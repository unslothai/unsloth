# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool definitions and executors for LLM tool calling: web search (DuckDuckGo), Python code
execution, and terminal commands."""

import ast
import codecs
import copy
from collections import deque
import fnmatch
import functools
import hashlib
from html.parser import HTMLParser
import json
import http.client
import os
from functools import partial
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

# Truncation notice cost, charged where the cut is decided.
from .context_window import _RESULT_NOTICE_RESERVE, compaction_receipt_field

# Window of the model serving THIS request (set by execute_tool). Unset falls back to the
# global probe, which is wrong for external providers.
_UNSET_CONTEXT_TOKENS = object()
_REQUEST_CONTEXT_TOKENS: ContextVar = ContextVar(
    "unsloth_request_context_tokens",
    default = _UNSET_CONTEXT_TOKENS,
)

# What the conversation has left, not the window size; None means unknown.
_REQUEST_RESULT_BUDGET: ContextVar = ContextVar(
    "unsloth_request_result_budget_tokens",
    default = None,
)

import uuid
import time
import urllib.parse
import urllib.request

from core.inference.mcp_image import (
    ATTACHED_IMAGE,
    image_input_mappings,
    image_mapping,
    public_tool,
    settle_image_call,
    strip_attached_image_note,
)
from core.inference.mcp_client import (
    MCP_TOOL_PREFIX,
    MCP_IMAGES_SENTINEL,
    TOOL_CACHE_INVALIDATING_FIELDS,
    cache_tools,
    call_tool_sync,
    get_cached_tools,
    in_failure_cooloff,
    is_studio_decisions,
    is_stdio,
    list_tools_async,
    oauth_client_kwargs,
    parse_server_headers,
    parse_stdio_command,
    probe_timeout,
    record_probe_failure,
    stdio_mcp_enabled,
    tool_ui_resource_uri,
    tool_visible_to,
)
from storage import mcp_servers_db
from utils.account_context import account_thread, current_account_id, is_owner_context
from utils.current_date_prompt_settings import strip_current_date_update_note
from utils import sandbox_memory_limit
from core.inference.tool_confinement import ToolConfinementUnavailable, account_confinement
from pathlib import Path
from utils.paths.storage_roots import RetiredAccountError, ensure_dir

from . import os_sandbox

from loggers import get_logger

logger = get_logger(__name__)

_EXEC_TIMEOUT = 300
_RAG_SEARCH_SLOT = threading.BoundedSemaphore(1)
_POLICY_OVERFETCH = 4
_DISABLE_DNS_PINNING_ENV = "UNSLOTH_STUDIO_DISABLE_DNS_PINNING"

RAG_SOURCES_SENTINEL = "\n__RAG_SOURCES__:"

# A search that produced nothing usable: not a tool error, so callers must test for these.
EMPTY_SEARCH_RESULTS = (
    "No results found.",
    "No results found within the website access limits.",
)
_DDGS_EMPTY_SWEEP = "No results found"
# Not "connection error": DNS failures and refused connections are not resets (#12638).
_DDGS_RESET_MARKERS = (
    "connection reset",
    "h2 connection driver error",
    "server disconnected",
    "broken pipe",
    "forcibly closed",  # Windows WSAECONNRESET (10054)
)
_DDGS_HTTP1_RETRY_LOCK = threading.Lock()

# Tier 2 runs only if tier 1 found nothing; engines in neither tier are never contacted.
_SEARCH_ENGINE_TIERS = (
    ("wikipedia", "brave", "duckduckgo", "mojeek", "startpage"),
    ("grokipedia", "google", "yahoo"),
)

# Module level so preexec_fn imports nothing in the forked child (deadlock risk).
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

# Pinned equal to _SANDBOX_MEDIA_TYPES by test_sandbox_files_and_storage_roots. No .svg (XSS),
# .html or .pdf.
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".avif"})


def _env_int(name: str, default: int) -> int:
    """Read an int env override; fall back to ``default`` on unset/garbage."""
    try:
        value = int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# Model-visible cap; the live UI stream cap is separate and higher.
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
        # POSIX synonym for `source`; matched at command position only.
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
# Words after which a command position starts; `coproc rm -rf x` really runs rm.
_SHELL_KEYWORDS_AS_SEP = frozenset(
    {"then", "do", "else", "elif", "if", "while", "until", "!", "coproc"}
)
# `coproc NAME` is valid only before a compound command; `{`/`(` are already separators.
_COPROC_COMPOUND_STARTERS = frozenset({"if", "while", "until", "for", "case", "select"})
_COPROC_NAME_RE = re.compile(r"^[A-Za-z_]\w*$")


def _is_coproc_name(tokens: "list[str]", index: int) -> bool:
    """Whether `tokens[index]` is the NAME of a `coproc NAME compound-command`, not a command word.

    Callers gate this on having just consumed the `coproc` KEYWORD, which the neighbouring tokens cannot decide:
    `echo coproc JOB if rm -f x` prints three words while `time coproc JOB if rm -f x` really deletes. They then
    READ the name like any other command word rather than skipping it, since shlex has already dropped the quotes
    and a forged `'{'` would otherwise hide whatever stands there; only command position carries past it.
    """
    return (
        index > 0
        and tokens[index - 1] == "coproc"
        and index + 1 < len(tokens)
        and tokens[index + 1] in _COPROC_COMPOUND_STARTERS
        and _COPROC_NAME_RE.match(tokens[index]) is not None
    )


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
# Unconsumed, `env -u FOO rm -rf x` would read as command `FOO`.
_WRAPPER_VALUE_FLAGS_BY_CMD = {
    "env": frozenset({"-u", "--unset"}),
    "stdbuf": frozenset({"-i", "--input", "-o", "--output", "-e", "--error"}),
    "timeout": frozenset({"-s", "--signal", "-k", "--kill-after"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "ionice": frozenset({"-c", "--class", "-n", "--classdata", "-p", "--pid"}),
    "xargs": frozenset(
        {"-I", "-L", "-P", "-d", "--delimiter", "-a", "--arg-file", "-n", "-s", "-E"}
    ),
    "chroot": frozenset({"--userspec", "--groups"}),
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
    "exec": frozenset({"-a"}),
    "setsid": frozenset(),
    "nohup": frozenset(),
}
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Prefixes that change command lookup or code loading (`LD_PRELOAD=x ls`, `PATH=. ls`).
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
        "LESSOPEN",
        "LESSCLOSE",
    }
)


# Entries that can shadow a real binary/module: absolute, home or parent escape.
_PATH_ENTRY_ESCAPES_RE = re.compile(r"(?:^|:)\s*(?:/|~|\$|[A-Za-z]:[\\/]|\.\.)")


def _env_assignment_is_unsafe(name: str, value: str = "") -> bool:
    """True if a NAME=value prefix affects command lookup/loading."""
    if name in _AUTO_UNSAFE_ENV_ASSIGN or name.startswith(("LD_", "DYLD_")):
        return True
    if name == "PATH":
        # PATH picks the binary, so any value counts (`PATH=. ls` runs ./ls).
        return True
    # Other search paths only shadow a module when the entry escapes the workdir.
    return name.endswith("PATH") and bool(_PATH_ENTRY_ESCAPES_RE.search(value))


# Read subcommands run silently; unrecognised ones still ask.
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
# awk can shell out via system() or pipes, so the program is screened.
_AWK_COMMANDS = frozenset({"awk", "gawk", "mawk", "nawk", "busybox-awk"})
_AWK_SHELL_ESCAPE_RE = re.compile(
    r"\bsystem\s*\(|\|\s*&?\s*[\"']\s*(?:/\S*/)?(?:sh|bash|zsh|ksh|dash|cmd)\b|"
    r"\bENVIRON\s*\[|\bprintf\s*\|"
)
# GNU sed `e` and `s///e` run shell commands.
_SED_COMMANDS = frozenset({"sed", "gsed", "ssed"})
# `w` absent: it takes the rest of the line as a filename.
_SED_SUBST_FLAGS = frozenset("0123456789gpiImMe")
# -e/-f/-l consume text; -i's suffix is attached only (`-ifoo` is not `-f oo`).
_SED_VALUE_FLAGS = "efl"
_SED_ATTACHED_VALUE_FLAGS = "i"
_SED_TEXT_ESCAPE_RE = re.compile(r"\\([\s\S])")
# Bare `$NAME` / `${NAME}` only; anything with an operator leaves the program UNREAD.
_PROGRAM_VAR_RE = re.compile(r"\$\{(\w+)\}|\$(\w+)")
# Unbraced name/positional/special params; any other `$` (sed's `$` address) is literal.
_UNBRACED_PARAM_RE = re.compile(r"\$(?:[A-Za-z_]\w*|[0-9]+|[@*#?$!-])")
# Arithmetic yields an integer, never a sed command; a digit stands in for it.
_ARITHMETIC_VALUE = "0"
# Per-invocation floor that keeps padded lines linear.
_MAX_SED_ARG_SCAN = 128
# Total sed arg budget per command line, split across its sed words.
_SED_SCAN_BUDGET = 200_000
# Bounded so `-exec env -exec env ...` padding cannot make the scan quadratic.
_MAX_EXEC_PREFIX_SCAN = 32
# Window quadrupled until the span closes, keeping many short substitutions linear.
_SUBSTITUTION_SPAN_STEP = 64
# A backslash and the char behind it: bash expands neither (`\$(` opens no substitution).
_ESCAPED_CHAR_STATE = "\\"
_WIN_CONDITIONAL_KEYWORDS = frozenset({"exist", "defined", "errorlevel", "cmdextversion", "not"})
_WIN_COMPARISON_OPS = frozenset({"==", "equ", "neq", "lss", "leq", "gtr", "geq"})
_FIND_EXEC_FLAGS = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
# Words after the terminator are find's next predicate, not CMD's.
_FIND_EXEC_TERMINATORS = frozenset({"+", ";", "\\;"})
# Quoted `';'` and `\;` reach find as the same word; `+` terminates only right after `{}`.
_FIND_EXEC_SEMICOLONS = frozenset({";", "\\;"})
# Only inside an action: outside one a quoted `';'` is a sed FILE operand; only a bare `;` ends it.

# Masked during a second lex so a quoted separator is told apart from a real one.
_SEPARATOR_CHARS = frozenset("".join(_SHELL_SEPARATORS))
# Any non-whitespace, non-quote, non-punctuation char keeps the same word split.
_QUOTED_SEPARATOR_MARK = "\x00"
_GLOB_CHARS = frozenset("*?[")
_QUOTED_GLOB_MARK = "\x01"
# A quoted redirection char is an ordinary word the command receives.
_REDIRECT_CHARS = frozenset("<>")
_QUOTED_REDIRECT_MARK = "\x02"
# Double quotes still expand, so only single-quoted and escaped states count.
_EXPANSION_CHARS = frozenset("$`")
_QUOTED_EXPANSION_MARK = "\x04"
# punctuation_chars glue runs like `|&` that match no separator entry; `{}` must stay a word.
_OPERATOR_TOKEN_CHARS = frozenset(";&|()`\n")
# The newline starts a new command in bash, where shlex sees only whitespace.
_COMMAND_POSITION_PUNCTUATION = ";&|()`\n"
# `&` splits off under punctuation_chars, so `2>&1` arrives as three tokens.
_REDIRECTION_RE = re.compile(r"^(?:\d+|&)?(?:<<<|<<-|<<|<>|>>|>\||<&|>&|<|>)")


def _punctuation_lexer(text: str, punctuation: str) -> "shlex.shlex":
    """A word lexer that hands back every one of ``punctuation`` as its own token.

    shlex counts a newline as whitespace, so asking for it in punctuation_chars alone yields
    nothing: it has to leave the whitespace set as well, or the separator bash sees at a line break
    never appears in the token stream.
    """
    lexer = shlex.shlex(text, posix = True, punctuation_chars = punctuation)
    lexer.whitespace_split = True
    if "\n" in punctuation:
        lexer.whitespace = lexer.whitespace.replace("\n", "")
    return lexer


def _looks_like_separator(token: str) -> bool:
    """Whether a lexed token is a shell operator rather than a word a command receives: a known
    separator, or a RUN of punctuation_chars characters, which is how bash builds `|&`, `;;` and
    `;&`."""
    if token in _SHELL_SEPARATORS:
        return True
    return bool(token) and not (set(token) - _OPERATOR_TOKEN_CHARS)


def _redirection_span(
    tokens: "list[str]",
    index: int,
    quoted: "frozenset[int]" = frozenset(),
    quoted_redirects: "frozenset[int]" = frozenset(),
) -> "tuple[int, ...]":
    """The token indexes one shell redirection at ``index`` occupies, or ``()``. The shell REMOVES a
    redirection before the command sees its arguments, so leaving the words in place made it the
    command's first operand (verified: `sed </dev/null '1e rm -f victim' input` runs). A detached
    target is claimed only when it is an ordinary word."""
    if tokens[index] == "&" and index + 1 < len(tokens) and tokens[index + 1][:1] in "<>":
        # `&>out.txt` splits in two; only a redirection may follow the `&` here.
        tail = _redirection_span(tokens, index + 1, quoted, quoted_redirects)
        return (index, *tail) if tail else ()
    if index in quoted_redirects:
        # Quoted, it is a word the command receives (e.g. a sed -f script file).
        return ()
    match = _REDIRECTION_RE.match(tokens[index])
    if not match:
        return ()
    if tokens[index][match.end() :]:
        return (index,)
    span = [index]
    nxt = index + 1
    if nxt >= len(tokens):
        return tuple(span)
    if tokens[nxt] in {"&", "|"}:
        # `2>&1` and `>|out.txt` arrive as three tokens.
        span.append(nxt)
        nxt += 1
    if nxt < len(tokens) and not (_looks_like_separator(tokens[nxt]) and nxt not in quoted):
        # The target goes to open(), not sed; only a bare operator is refused.
        span.append(nxt)
    return tuple(span)


_TEST_BUILTINS = frozenset({"[", "[[", "]", "]]"})


def _is_unresolved_command_glob(base: str) -> bool:
    """Whether a command word is a glob bash expands to some other name (`/bin/r[m]` runs rm). A
    pattern with no literal character is not one, and the test builtins are not patterns."""
    if base in _TEST_BUILTINS or not any(ch in base for ch in "*?["):
        return False
    return any(ch.isalnum() for ch in base)


def _blocked_matching_glob(base: str) -> "set[str]":
    """Blocked command names a command-position glob can expand to."""
    if not _is_unresolved_command_glob(base):
        return set()
    return {name for name in _BLOCKED_COMMANDS if fnmatch.fnmatchcase(name, base)}


def _is_sed_command(base: str) -> bool:
    """Whether a command word runs sed: an exact name, or a command-position GLOB that could expand
    to one, since bash resolves `/usr/bin/s[e]d` to sed after this scan. Fail closed."""
    if base in _SED_COMMANDS:
        return True
    return _is_unresolved_command_glob(base) and any(
        fnmatch.fnmatchcase(name, base) for name in _SED_COMMANDS
    )


def _sed_short_flag(token: str) -> "tuple[str, str] | None":
    """The first value-taking short option in a sed flag cluster, as ``(letter, text glued after
    it)``, or ``None``. The scan stops there because the rest of the token is that option's
    value: `-ifoo` is -i with suffix "foo", not an attached -f."""
    if not token.startswith("-") or token.startswith("--"):
        return None
    for index, ch in enumerate(token[1:]):
        if ch in _SED_VALUE_FLAGS or ch in _SED_ATTACHED_VALUE_FLAGS:
            return ch, token[index + 2 :]
    return None


def _sed_long_flag(name: str) -> str:
    """Which value-taking sed long option ``--name`` is: "e", "f", "l" or "". getopt allows
    unambiguous abbreviations, so --e/--ex are --expression and --fi upwards is --file (--f is
    ambiguous with --follow-symlinks)."""
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
    """Whether the long option ``name`` puts sed in a mode that REFUSES to shell out. --sandbox
    disables e/r/w and --posix drops the GNU extensions `e` belongs to, so a script COMPILED
    under either aborts the run and its payload is inert; which scripts that covers depends on
    where the flag sits (see _sed_invocation). Only unambiguous abbreviations count."""
    if len(name) >= 4 and "--sandbox".startswith(name):
        return True
    return len(name) >= 3 and "--posix".startswith(name)


def _sed_scan_limit(sed_words: int) -> int:
    """How many argument tokens ONE sed invocation may walk looking for its script. A lone sed gets
    the whole budget, so padding cannot push the script out of view; a line packed with sed words
    falls back to the floor, which keeps the walk linear (39s against 3s)."""
    if sed_words <= 1:
        return _SED_SCAN_BUDGET
    return max(_MAX_SED_ARG_SCAN, _SED_SCAN_BUDGET // sed_words)


# The script arrives on stdin, so "no program found" is ignorance, not safety.
_SED_STREAM_PROGRAM_SOURCES = frozenset({"-", "/dev/stdin", "/dev/fd/0"})


def _sed_program_source_is_stream(value: str) -> bool:
    """Whether an `-f` operand reads the script from a stream this scan cannot follow. A named file
    stays out: it is documented residue rather than something to fail on. A process substitution
    counts, since `sed -f <(printf 'e rm -f victim') input` really runs rm, and the lexer splits
    that operand at the `(`."""
    if value in _SED_STREAM_PROGRAM_SOURCES or value.startswith("/dev/fd/"):
        return True
    return value[:1] in "<>"


def _end_program_source(programs: "list[str]", exec_disabled: bool) -> None:
    """Close the script source the pieces collected so far belong to, by appending the blank line
    the join needs. A source BOUNDARY ends any line continuation open across it, so a trailing
    `a\\` appends a blank line instead of swallowing the next source's first line (verified on GNU
    sed 4.9)."""
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
    """The sed invocation whose command word sits at ``start``, as ``(program alternatives, unread,
    live_program)``.

    sed joins its -e values with newlines, so `sed -e '1a\' -e 'e rm -rf x'` appends a line instead
    of executing it and the pieces are judged together. With no -e or -f the first positional is the
    script.

    --sandbox / --posix abort at COMPILE time, and sed compiles each -e as it is parsed while the
    positional waits for the whole option list, so the flag suppresses exactly the scripts written
    after it (verified on GNU sed 4.9). One written after the POSITIONAL suppresses only while
    getopt permutes, and POSIXLY_CORRECT turns that off from outside the command text, so it is not
    read as suppressing. `--` is honoured.

    ``unread`` says the program is at best a PREFIX of the real one, so an empty result proves
    nothing and callers fail closed on it.

    ``stops`` and ``skips`` are token INDEXES, not text: where the invocation ends and which words
    are a redirection the shell removes. Both distinctions need the original quoting, which the text
    has lost. A skip yields to a pending -e/-f/-l value, since that word is sed's.
    """
    programs: "list[str]" = []
    first_positional = ""
    positional_disabled = False
    positional_globbed = False
    positional_live = False
    # A program flag ahead of the positional makes it an input file; behind it only under getopt
    # permutation, which POSIXLY_CORRECT disables (GNU sed 4.9).
    program_flag_before_positional = False
    # After a mode flag every compiled script is inert; live pieces are always a prefix.
    exec_disabled = False
    end_of_options = False
    value_pending = ""
    hit_separator = False
    stream_program = False
    glob_program = False
    live_program = False
    window = tokens[start + 1 : start + 1 + limit]
    for offset, token in enumerate(window):
        if start + 1 + offset in stops:
            hit_separator = True
            break
        if start + 1 + offset in skips:
            # Checked ahead of the pending value: a redirection there is removed and the value follows it.
            continue
        if value_pending:
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
            found = _sed_short_flag(token)
            if found is None:
                continue
            letter, attached = found
            if letter in _SED_ATTACHED_VALUE_FLAGS:
                # -i's suffix never takes the next token.
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
            # Which script sed compiles depends on permutation; keep both as alternatives, not one program.
            joined.append(first_positional)
    scan_overflowed = not hit_separator and len(tokens) > start + 1 + limit
    # A pending -f value means the operand was never read: the program is unknown, not absent.
    joined = [piece.replace(_ANSI_C_NEWLINE_MARK, "\n") for piece in joined]
    unread = scan_overflowed or stream_program or glob_program or value_pending == "f"
    return joined, unread, live_program


def _sed_text(text: str) -> str:
    """Unescape one sed text argument the way read_text does: every backslash drops away and the
    character behind it stays, so `e touch MARK\\ER` runs MARKER."""
    return _SED_TEXT_ESCAPE_RE.sub(r"\1", text).strip()


def _sed_exec_payloads(program: str) -> "list[str]":
    """Shell payloads a sed program executes, in order.

    `e COMMAND` runs COMMAND. A bare `e` and the `s///e` flag run the pattern space, which only
    exists at run time, so they yield an EMPTY payload: executes, but nothing to screen. An empty
    list means it only edits text.

    The walk skips every region where an `e` is data (regexes, replacements, a/i/c text, r/w
    filenames, b/t labels, comments).
    """
    payloads: "list[str]" = []
    n = len(program)

    def _end_of_line(pos: int) -> int:
        end = program.find("\n", pos)
        return n if end < 0 else end

    def _end_of_text(pos: int) -> int:
        # A trailing backslash carries `e`/`a`/`i`/`c` text onto the next line.
        while pos < n and program[pos] != "\n":
            pos += 2 if program[pos] == "\\" else 1
        return min(pos, n)

    def _skip_bracket(pos: int) -> int:
        # In a bracket expression the delimiter is data; a leading `]` is literal.
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
        # Brackets apply to regex halves only.
        while pos < n and program[pos] != delim:
            if program[pos] == "\\":
                pos += 2
            elif brackets and program[pos] == "[":
                pos = _skip_bracket(pos)
            else:
                pos += 1
        return pos + 1

    def _skip_address(pos: int) -> int:
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
                i += 1
                while i < n and program[i].isdigit():
                    i += 1
            else:
                i = _skip_address(i)
        while i < n and program[i] in " \t!":
            i += 1
        if i >= n:
            break
        cmd, i = program[i], i + 1
        if cmd == "e":
            # Payload ends at an unescaped newline; `;` inside is shell text.
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
            i = _end_of_text(i)
        elif cmd in "rRwW":
            i = _end_of_line(i)
        elif cmd in "btT:v":
            while i < n and program[i] not in ";\n}":
                i += 1
    return payloads


def _assignment_bindings(
    tokens: "list[str]", quoted: "frozenset[int]" = frozenset()
) -> "list[tuple[int, str, str | None]]":
    """Every `NAME=value` word as ``(token index, name, value)``, in the order the shell performs the
    assignments.

    An ordered LIST, not a map, because bash uses the binding performed most recently BEFORE the
    reference: first-wins let `p='1,3p'; p='1e rm -f victim'; sed "$p" input` read as `1,3p` while
    rm really runs. The index rides along so _bindings_before can drop assignments that only happen
    after the sed.

    A non-literal value is recorded as ``None``, which CLEARS the name rather than leaving a stale
    earlier one standing.

    Only a word that really changes SHELL state counts: an assignment-shaped ARGUMENT, one in a
    subshell and one used as a command's environment prefix all leave `$p` alone, and recording them
    overwrote a payload with a value bash never assigned. A conditional one after `&&` may or may
    not run, so it is UNRESOLVED instead.
    """
    bindings: "list[tuple[int, str, str | None]]" = []
    pending: "list[tuple[int, str, str | None]]" = []
    at_command = True
    depth = 0
    conditional = False
    function_body = 0
    saw_parens = False
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
            bindings.extend(pending)
            pending = []
            saw_parens = set(token) <= {"(", ")"} and ")" in token
            depth = max(0, depth + token.count("(") - token.count(")"))
            conditional = "&&" in token or "||" in token
            at_command = True
            continue
        if function_body and _ASSIGNMENT_RE.match(token):
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
            # Assignments before a command word are that child's environment, not the shell's.
            pending = []
            at_command = False
    bindings.extend(pending)
    return bindings


def _bindings_before(
    bindings: "list[tuple[int, str, str | None]]", cursor: int, limit: int, env: "dict[str, str]"
) -> int:
    """Fold into ``env`` every binding at a token index below ``limit``, starting at ``cursor``, and
    return the cursor to pass in next time. Later bindings overwrite earlier ones. Seds are
    visited left to right, so the cursor only moves forward and the whole line costs ONE walk of
    the binding list."""
    while cursor < len(bindings) and bindings[cursor][0] < limit:
        _index, name, value = bindings[cursor]
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
        cursor += 1
    return cursor


def _resolve_program_vars(program: str, env: "dict[str, str]") -> str:
    """``program`` with each `$NAME` / `${NAME}` replaced by its assigned value. A sed script held
    in a variable is only a program once the reference is resolved, and only in a pass that KEEPS
    the quoted newline: the blanket newline pass turns the value into one long sed comment. An
    unassigned name is left as written."""
    return _PROGRAM_VAR_RE.sub(lambda m: env.get(m.group(1) or m.group(2), m.group(0)), program)


def _sed_program_variants(program: str, env: "dict[str, str]") -> "list[str]":
    """The sed program as written, plus the variable-resolved and arithmetic-collapsed forms. All
    are screened, because any spelling can be the one holding the `e`."""
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
    """One expansion, keyed so the raw-command spelling and the post-lex one compare equal. Only the
    escaping differs between them, so it is dropped."""
    return text.replace("\\", "")


def _sed_program_unresolved(variants: "list[str]", live: "set[str]") -> bool:
    """Whether NO spelling of the sed program is one this scan actually READ, because every one still
    holds an expansion bash would rewrite.

    The program is knowable only when each expansion reduces to text. The parameter transformations
    (`${p%y}`, `${p/a/b}`, `${p:-z}`, ...) are not modelled one at a time; an unread program is
    UNKNOWN and the auto gate asks, which makes every unmodelled form safe by default rather than a
    way past.

    Only expansions the shell RUNS count, and only where they land in the PROGRAM, so one the
    program merely quotes, an escaped one and one in a FILE operand are all left running.
    """
    if not live:
        return False
    # shlex removes escaping, so key both spellings without backslashes (fails closed).
    keys = {_expansion_key(found) for found in live}
    return not any(
        all(_expansion_key(found) not in keys for found in _shell_expansions(variant, quoted = False))
        for variant in variants
    )


def _quoted_separator_indexes(text: str, tokens: "list[str]", punctuation: str) -> "frozenset[int]":
    """Indexes of ``tokens`` that only LOOK like a shell separator because the quoting has been
    stripped off them.

    shlex hands back the identical token `;` for a real separator and for a quoted `';'` a command
    receives as data, so `sed -n ';' -e '1e rm -f victim' input` looked like a sed that had already
    ended (verified on GNU sed 4.9: it runs rm).

    Told apart by masking every separator character the shell QUOTES and lexing a second time. Only
    those characters change, and each inside the word it already belonged to, so the two token lists
    line up; the alignment is asserted by the length check.
    """
    if not any(_looks_like_separator(token) for token in tokens):
        return frozenset()
    if _QUOTED_SEPARATOR_MARK in text:
        return frozenset()
    states = _shell_quote_states(text)
    masked = "".join(
        _QUOTED_SEPARATOR_MARK if char in _SEPARATOR_CHARS and states[index] else char
        for index, char in enumerate(text)
    )
    if _QUOTED_SEPARATOR_MARK not in masked:
        return frozenset()
    try:
        marked = list(_punctuation_lexer(masked, punctuation))
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
    """``tokens`` re-lexed with every one of ``chars`` the QUOTING made literal replaced by
    ``mark``, or ``None`` when the two lexes do not line up. Each replacement stays inside the
    word it already belonged to, so the second lex yields the same words; the alignment is
    asserted by the length check."""
    if not any(char in chars for char in text) or mark in text:
        return None
    states = _shell_quote_states(text)
    masked = "".join(
        mark if char in chars and states[index] else char for index, char in enumerate(text)
    )
    try:
        marked = list(_punctuation_lexer(masked, punctuation))
    except ValueError:
        return None
    return marked if len(marked) == len(tokens) else None


def _quoted_redirection_indexes(
    text: str, tokens: "list[str]", punctuation: str
) -> "frozenset[int]":
    """Indexes of ``tokens`` that only LOOK like a redirection because the quoting has been stripped
    off them. A QUOTED redirection is a word the shell hands the command: `sed -f '>prog' -e '1e
    rm -f victim' input` takes `>prog` as the script FILE and really runs the payload. Decided on
    the operator the token OPENS with, so `2>'/dev/null'` stays a redirection."""
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
    """Indexes of ``tokens`` holding an expansion the shell really PERFORMS. Live expansions are
    collected over the whole command, so matching a sed program against them by text alone
    attributed another command's expansion to a program that merely spells the same thing; this
    supplies the missing occurrence. Double quoting is deliberately not literal: `sed "$p" f`
    expands and must stay in."""
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
        marked = list(_punctuation_lexer(masked, punctuation))
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
    """Indexes of ``tokens`` holding a pathname-expansion metacharacter the shell will EXPAND,
    rather than one the quoting made literal. bash expands after this scan, so in a directory
    holding a file named `1e rm -f victim`, `sed *` hands sed that filename as its script and
    really runs rm. The quoted spellings a sed program uses must stay readable. Told apart by
    masking and re-lexing."""
    if not any(char in _GLOB_CHARS for char in text) or _QUOTED_GLOB_MARK in text:
        return frozenset()
    states = _shell_quote_states(text)
    masked = "".join(
        _QUOTED_GLOB_MARK if char in _GLOB_CHARS and states[index] else char
        for index, char in enumerate(text)
    )
    try:
        marked = list(_punctuation_lexer(masked, punctuation))
    except ValueError:
        return frozenset()
    if len(marked) != len(tokens):
        return frozenset()
    return frozenset(
        index for index, token in enumerate(marked) if any(char in _GLOB_CHARS for char in token)
    )


def _xargs_replacement(tokens: "list[str]", start: int, end: int) -> str:
    """The placeholder the xargs word at ``start`` substitutes into the command words behind it, or
    "" when it replaces nothing. GNU xargs takes it attached (`-I{}`), as the next word or after
    an `=`; `-i` and a bare `--replace` default to `{}`."""
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
    """Whether an xargs is the one deciding what program its sed runs. xargs appends the words it
    reads on stdin, and with -I substitutes them into the words already there, so the program
    need not be in the command TEXT at all (`printf '1e rm -f victim\0input\0' | xargs -0 sed`
    runs rm), and the sed fails closed. The ordinary idioms are untouched, since their program is
    right there and the placeholder stands where the FILE goes."""
    if not program.strip():
        return True
    placeholder = _xargs_replacement(tokens, xargs, sed)
    return bool(placeholder) and placeholder in program


def _sed_program_is_a_placeholder(program: str) -> bool:
    """Whether the whole sed program is a token another tool REWRITES before sed starts. find
    replaces `{}` with the pathname it found, so with a file named `1e rm -f victim` a `find ...
    -exec xargs sed {} +` really runs rm while `{}` read as an already-known program. A `{}`
    among the FILE operands is not the program and is untouched."""
    return program.strip() == "{}"


def _forwards_exec_flags(base: str) -> bool:
    """Whether a command word runs a tool whose `-exec` / `-x` options hand the words behind them to
    a child command. Exact names, plus any command-position GLOB that could expand to one."""
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
    """``(exec-flag indexes, invocation-stop indexes, redirection indexes)`` for one token list, in a
    single left-to-right pass.

    An exec-flag index is a `find`/`fd` option whose following words are a COMMAND that tool runs.
    Recognised only while a find/fd word the shell really RUNS is in scope: those letters belong to
    too many other tools, so `grep -x rm file` must not have rm hard-blocked.

    A stop index ends a sed invocation: a separator the shell PERFORMS, or the `;` / `{} +` closing
    an open exec action. Outside an action those are ordinary operands.

    A redirection index is a word the shell consumes and never hands to the command. Taken FIRST, so
    the `&` in `sed 2>&1 '1e rm -f victim' input` reads as part of that redirection rather than as
    the end of the invocation.
    """
    exec_flags: "set[int]" = set()
    stops: "set[int]" = set()
    redirects: "set[int]" = set()
    forwarding = False
    in_action = False
    at_command = True
    wrapper = ""
    skip_operand = False
    coproc_kw = False
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
        after_coproc = coproc_kw
        coproc_kw = False
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
            # find ends the batch form only at `{} +`; other `+` words go to the child.
            stops.add(here)
            in_action = False
            continue
        if forwarding and token == "--" and not in_action:
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
            # fd also takes the command attached: `-xrm` (fdfind 9.0.0).
            exec_flags.add(here)
            in_action = True
            continue
        if at_command and token in _SHELL_KEYWORDS_AS_SEP:
            coproc_kw = token == "coproc"
            continue
        if skip_operand:
            skip_operand = False
            continue
        if token.startswith("-") or _ASSIGNMENT_RE.match(token):
            # A wrapper option's separate value token precedes the command (`env -u FOO find`).
            skip_operand = token in _WRAPPER_VALUE_FLAGS_BY_CMD.get(wrapper, frozenset())
            continue
        if wrapper and token.lstrip("-").isdigit():
            continue
        base = os.path.basename(token.strip(";&|()`{}")).lower()
        coproc_name_here = after_coproc and _is_coproc_name(tokens, here)
        if at_command and base in _COMMAND_PREFIXES and not coproc_name_here:
            wrapper = base
            continue
        if at_command and _forwards_exec_flags(base):
            # Only a find/fd at command position forwards exec flags (`echo fd -x rm` is harmless).
            forwarding = True
        at_command = coproc_name_here
        wrapper = ""
    return frozenset(exec_flags), frozenset(stops), frozenset(redirects)


def _win_switch(token: str) -> str:
    """Collapse a Git Bash `//x` switch to the `/x` cmd.exe actually receives."""
    return token[1:] if token.startswith("//") else token


# `start` launches its argument; value-taking switches eat a token.
_START_SWITCHES_WITH_VALUE = {"/d", "/node", "/affinity", "/machine"}

# Matched in full so a program path (/bin/bash) is never skipped as a switch.
_CMD_SWITCH_RE = re.compile(r"/[a-zA-Z](?::[\w.]+)?")

# Matched by name: MSYS rewrites POSIX paths into slash-led program paths.
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
    """True when START would read ``token`` as its window title, not the program. The cmd lexer
    keeps the quote marks, so a title still arrives quoted; the posix lexer leaves two spellings
    a bare program name cannot have, the empty ``start ""`` idiom and a title containing
    whitespace. A single-word posix title is indistinguishable from a program name and is
    deliberately not guessed."""
    return (
        token == ""
        or any(char.isspace() for char in token)
        or (len(token) >= 2 and token[0] == '"' and token[-1] == '"')
    )


# Built once: the set never mutates and rebuilding per call was a measurable cost.
def _blocked_word_re(assignment_prefixes: bool):
    """The command-position backstop, with and without the assignment-prefix step.

    The backstop has no quoting model, so anything it steps over it steps over inside quotes too.
    Skipping assignment prefixes is only needed when the lex raised and the token walk never
    happened; running it when the walk succeeded refused `echo '; A=1 rm -rf x'`, which bash runs
    as one `echo` (checked: the file survives). A quoted value may hold the rest of the line
    (`p='1e rm -f victim'`), which is a binding the sed screen resolves, so those are deliberately
    not stepped over either way.
    """
    if not _BLOCKED_COMMANDS:
        return None
    return re.compile(
        # No `coproc` here: without command-position context it refused `grep coproc rm file`.
        r"(?:^|[;&|`\n(]\s*|[$]\(\s*|<\(\s*)"
        + (r"(?:[A-Za-z_]\w*=[^\s'\"]*\s+)*" if assignment_prefixes else r"")
        + r"(?:[\w./\\-]*/|[a-zA-Z]:[/\\][\w./\\-]*)?"
        + r"("
        + "|".join(re.escape(w) for w in sorted(_BLOCKED_COMMANDS))
        + r")"
        + r"(?:\.(?:exe|com|bat|cmd))?\b"
    )


_BLOCKED_WORD_RE = _blocked_word_re(False)
_BLOCKED_WORD_RE_WITH_ASSIGNMENTS = _blocked_word_re(True)


def _join_escaped_newlines(text: str) -> str:
    """Remove a backslash-newline where the shell removes it, and only there.

    The shell strips the pair before it reads a command, so the line break is not a boundary and
    the words either side belong to one command: `echo hi \\<newline>A=1 rm -rf x` is one `echo`.
    Inside SINGLE quotes it strips nothing, and that difference is load-bearing:
    `sed -n '1e touch a\\<newline>rm -f victim' f` continues the executed payload onto the next
    line, so joining there would drop a command that really runs. An escaped backslash consumes
    both characters, which leaves a following newline standing, as the shell does. The pair is
    removed rather than replaced, since it can fall inside a word: checked against the same
    bash, `to\\<newline>uch f` runs `touch`.

    Only a backslash-LF continues. Checked against bash 5.2.21: a backslash before CRLF escapes
    the CARRIAGE RETURN, so the newline still starts a command and `echo hi \\<CRLF>rm -rf x`
    really runs `rm`. Joining all three characters made that read as one `echo`.

    A comment strips nothing either. Inside an unquoted `#` comment the backslash is comment
    TEXT, so the newline still ends the comment and starts a command: `echo ok # comment
    \\<newline>rm -rf ./build` really runs `rm`, and joining made it one comment that the
    later lex then discarded whole. `#` opens a comment only at the start of a word, so
    `ab#cd` is an ordinary word and `"a # b"` is quoted text; both were checked against the
    same bash.

    A `$(...)` substitution is parsed in a FRESH quoting context even inside double quotes, so
    the state is pushed at `$(` and popped at the matching `)`. Checked against bash 5.2.21:
    `echo "$(echo hi # c \\<newline>rm -f victim<newline>)"` really runs `rm`, because the `#`
    opens a comment in there and the backslash is comment text; carrying the outer `in_double`
    in made the pair look like a continuation and the `rm` vanished. The same probe confirms the
    other half: single quotes work in there (`"$(echo 'a \\<newline>rm -f victim')"` strips
    nothing and `rm` does NOT run), and a `)` inside them does not end the substitution.
    Backticks are left alone, since the same probe shows `#` does not open a comment inside
    `"`...`"`.
    """
    if "\\\n" not in text:
        # Nothing to join: return the input after one C-level scan.
        return text
    out: list[str] = []
    i, n = 0, len(text)
    in_single = in_double = in_comment = False
    # Quote state per enclosing `$(`, plus open `(` groups inside it: a subshell's `)` must not
    # restore the outer state.
    substitutions: "list[tuple[bool, bool, bool]]" = []
    group_depth = 0
    group_depths: list[int] = []
    closed_substitution = False
    # Open `case` count per substitution: a case pattern's `)` must not close the substitution.
    case_depth = 0
    case_depths: list[int] = []
    # Open `${...}` count per substitution: a `)` inside is part of the expansion.
    brace_depth = 0
    brace_depths: list[int] = []
    word = ""
    # `esac` closes a case only at command position; `case` counts anywhere (errs towards inside).
    at_command_position = True
    word_at_command_position = True
    while i < n:
        ch = text[i]
        was_substitution_close, closed_substitution = closed_substitution, False
        if not in_single and not in_double and not in_comment:
            if ch.isalpha() or ch == "_":
                if not word:
                    word_at_command_position = at_command_position
                word += ch
            else:
                if word == "case":
                    case_depth += 1
                elif word == "esac" and word_at_command_position and case_depth:
                    case_depth -= 1
                if word:
                    at_command_position = False
                    word = ""
                if ch in ";&|\n(":
                    at_command_position = True
                elif not ch.isspace():
                    at_command_position = False
        if not in_single and not in_comment and ch == "$" and text[i + 1 : i + 2] == "{":
            brace_depth += 1
            out.append("${")
            i += 2
            continue
        if ch == "}" and brace_depth and not in_single and not in_comment:
            brace_depth -= 1
            out.append(ch)
            i += 1
            continue
        if not in_single and not in_comment and ch == "$" and text[i + 1 : i + 2] == "(":
            substitutions.append((in_single, in_double, in_comment))
            group_depths.append(group_depth)
            case_depths.append(case_depth)
            brace_depths.append(brace_depth)
            group_depth = case_depth = brace_depth = 0
            in_single = in_double = in_comment = False
            out.append("$(")
            i += 2
            continue
        if ch == "(" and not in_single and not in_double and not in_comment and not brace_depth:
            group_depth += 1
            out.append(ch)
            i += 1
            continue
        if ch == ")" and not in_single and not in_double and not in_comment and not brace_depth:
            if group_depth:
                group_depth -= 1
            elif case_depth:
                pass
            elif substitutions:
                in_single, in_double, in_comment = substitutions.pop()
                group_depth = group_depths.pop()
                case_depth = case_depths.pop()
                brace_depth = brace_depths.pop()
                # A substitution's close stays in the word: a `#` after it is no comment (bash 5.2).
                out.append(ch)
                i += 1
                closed_substitution = True
                continue
            out.append(ch)
            i += 1
            continue
        if in_comment:
            if ch == "\n":
                in_comment = False
            out.append(ch)
            i += 1
            continue
        if in_single:
            if ch == "'":
                in_single = False
            out.append(ch)
            i += 1
            continue
        if (
            ch == "#"
            and not in_double
            and not was_substitution_close
            and (not out or out[-1].isspace() or out[-1] in ";&|()")
        ):
            in_comment = True
            out.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "\n":
                # Removed, not replaced by a space: `r\<newline>m -rf x` runs `rm`.
                i += 2
                continue
            out.append(ch)
            out.append(nxt)
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = True
        elif ch == '"':
            in_double = not in_double
        out.append(ch)
        i += 1
    return "".join(out)


def _find_blocked_commands(command: str, posix: "bool | None" = None) -> set[str]:
    """Detect blocked commands at shell command position only.

    A token is at command position if it is the first token, or follows a shell separator /
    brace-group opener / new-command keyword, or a command-prefix wrapper like `env` / `time` /
    `xargs`. Tokens in argument position pass through. Also scans `find ... -exec CMD` and recurses
    into bash -c / cmd /c. ``posix`` names the dialect of the shell that will run the command; None
    keeps the host default (_shell_is_posix).
    """
    blocked: set[str] = set()

    # Decode ANSI-C quoting first so `$'ssh'` is still detected at command position.
    command = _decode_ansi_c(command, keep_one_word = True)

    # bash removes backslash-newline before reading a command; join here so the token walk and the
    # backstop read the same text.
    command = _join_escaped_newlines(command)

    # Keyed to the shell that will run this, not the OS: a Windows bash still splits on `;`.
    lexed_posix = _shell_is_posix() if posix is None else posix
    try:
        if not lexed_posix:
            tokens = shlex.split(command, posix = False)
        else:
            tokens = list(_punctuation_lexer(command, _COMMAND_POSITION_PUNCTUATION))
    except ValueError:
        tokens = command.split()
        lexed_posix = False
        lex_raised = True
    else:
        lex_raised = False
    # The cmd lexer and split() fallback have no quoting model and report nothing.
    quoted_separators = (
        _quoted_separator_indexes(command, tokens, _COMMAND_POSITION_PUNCTUATION)
        if lexed_posix
        else frozenset()
    )
    quoted_redirects = (
        _quoted_redirection_indexes(command, tokens, _COMMAND_POSITION_PUNCTUATION)
        if lexed_posix
        else frozenset()
    )
    exec_flag_indexes, invocation_stops, redirect_indexes = _exec_scan_layout(
        tokens, quoted_separators, quoted_redirects
    )
    glob_indexes: "frozenset[int] | None" = None

    def _token_basename(tok: str) -> str:
        tok = tok.strip(";&|()`{}")
        base = os.path.basename(tok).lower()
        stem, ext = os.path.splitext(base)
        if ext in {".exe", ".com", ".bat", ".cmd"}:
            base = stem
        return base

    def _exec_child_index(start: int) -> "tuple[int, bool]":
        """The command a `find -exec` actually runs, as ``(index, overflowed)``; the index is -1 when
        the action holds no command word at all.

        Command prefixes forward to their target, so `-exec env sed ...` runs sed. Wrapper flags,
        assignment prefixes and duration operands are stepped over, and a wrapper option taking a
        SEPARATE value consumes it too, else that value reads as the command. The hop is bounded so
        `-exec env -exec env ...` cannot make this quadratic.

        ``overflowed`` says the bound ran out with words still ahead. That is NOT the same as
        finding nothing, and reporting both as "no child" let a long enough chain read as safe:
        `-exec` + 33 `env` + `rm -f victim ;` really deletes.
        """
        i, steps, wrapper = start, 0, ""
        while i < len(tokens) and steps < _MAX_EXEC_PREFIX_SCAN:
            token = tokens[i]
            if token in _SHELL_SEPARATORS or token in _FIND_EXEC_TERMINATORS:
                return -1, False
            steps += 1
            if wrapper and token in _WRAPPER_VALUE_FLAGS_BY_CMD.get(wrapper, frozenset()):
                # Option and operand consumed in one step, since the budget bounds steps per -exec.
                i += 2
                continue
            if wrapper and (
                token.startswith("-") or _ASSIGNMENT_RE.match(token) or token.lstrip("-").isdigit()
            ):
                i += 1
                continue
            base = _token_basename(token)
            if base in _COMMAND_PREFIXES:
                wrapper = base
                i += 1
                continue
            return i, False
        # Stopping on the bound with words still ahead means the child is UNREAD.
        return -1, steps >= _MAX_EXEC_PREFIX_SCAN and i < len(tokens)

    expect_command = True
    prefix_pending = False
    prefix_command = ""
    skip_operand = False
    sed_indexes: "list[int]" = []
    sed_xargs: "dict[int, int]" = {}
    xargs_index = -1
    coproc_kw = False
    if_condition = False
    skip_tokens = 0
    for token_index, token in enumerate(tokens):
        after_coproc = coproc_kw
        coproc_kw = False
        if skip_tokens:
            skip_tokens -= 1
            continue
        if skip_operand:
            skip_operand = False
            continue
        if expect_command and token.lower() == "if":
            # cmd's IF is case-insensitive.
            if_condition = True
            prefix_pending = False
            prefix_command = ""
            xargs_index = -1
            continue
        if if_condition and expect_command:
            low = token.lower()
            if low in {"/i", "not"}:
                continue
            if_condition = False
            following = tokens[token_index + 1].lower() if token_index + 1 < len(tokens) else ""
            if following in _WIN_COMPARISON_OPS:
                skip_tokens = 2
                continue
            if "==" in token.strip("="):
                continue
        if expect_command and token.lower() in _WIN_CONDITIONAL_KEYWORDS:
            skip_operand = token.lower() != "not"
            continue
        if prefix_pending and token == "-a":
            skip_operand = True
            continue
        if token_index in redirect_indexes:
            # Redirections leave command position unchanged: `> out.txt rm -rf victim` deletes.
            continue
        # A quoted operator is data, not a separator. cmd's FOR ... DO is case-insensitive.
        if (_looks_like_separator(token) and token_index not in quoted_separators) or (
            token.lower() in _SHELL_KEYWORDS_AS_SEP and expect_command
        ):
            coproc_kw = expect_command and token == "coproc"
            expect_command = True
            prefix_pending = False
            prefix_command = ""
            xargs_index = -1
            continue
        if token.startswith("-"):
            # Consume a wrapper option's separate value, else it reads as the command (`env -u PATH rm`).
            if prefix_pending and token in _WRAPPER_VALUE_FLAGS_BY_CMD.get(
                prefix_command, frozenset()
            ):
                skip_operand = True
                continue
            # Keep expect_command while a wrapper prefix awaits its command.
            if not prefix_pending:
                expect_command = False
            continue
        if not expect_command:
            continue
        if _REDIR_PREFIX_RE.match(token):
            continue
        if _ASSIGNMENT_RE.match(token):
            continue
        if prefix_pending and token.lstrip("-").isdigit():
            continue
        coproc_name_here = after_coproc and _is_coproc_name(tokens, token_index)
        base = _token_basename(token)
        if _is_sed_command(base):
            sed_indexes.append(token_index)
            if xargs_index >= 0:
                sed_xargs[token_index] = xargs_index
        if base in _BLOCKED_COMMANDS:
            blocked.add(base)
        else:
            blocked |= _blocked_matching_glob(base)
        # Wrappers consume one command; the privilege wrapper is also in _BLOCKED_COMMANDS.
        if base in _COMMAND_PREFIXES and not coproc_name_here:
            if base == "xargs" and xargs_index < 0:
                xargs_index = token_index
            prefix_pending = True
            prefix_command = base
            continue
        expect_command = coproc_name_here
        prefix_pending = False
        prefix_command = ""
        xargs_index = -1

    # An alias body runs when invoked, so scan it as a command.
    for i, tok in enumerate(tokens):
        if _token_basename(tok) != "alias":
            continue
        for nxt in tokens[i + 1 :]:
            if nxt in _SHELL_SEPARATORS:
                break
            _name, _sep, _value = nxt.partition("=")
            if _sep and _value:
                blocked |= _find_blocked_commands(_value, posix = posix)

    # find -exec/-execdir and fd -x/-X/--exec/--exec-batch all invoke CMD directly.
    for i, tok in enumerate(tokens):
        # `fd --exec=rm`: the attached value is command position.
        attached = ""
        if tok[:2] in {"-x", "-X"} and len(tok) > 2 and i in exec_flag_indexes:
            attached = tok[2:].strip("\"'")
        elif "=" in tok and tok.split("=", 1)[0] in _ATTACHED_EXEC_FLAGS:
            attached = tok.split("=", 1)[1].strip("\"'")
        if attached:
            attached_base = _token_basename(attached.split()[0])
            if _is_sed_command(attached_base):
                # Screen the forwarded sed's program from the flag; fd 9 runs nothing here, but forwarding
                # spellings would otherwise be a free pass.
                sed_indexes.append(i)
            if attached_base in _BLOCKED_COMMANDS:
                blocked.add(attached_base)
            else:
                blocked |= _blocked_matching_glob(attached_base)
        if i in exec_flag_indexes and i + 1 < len(tokens):
            # A wrapper is a command itself and a step to another; check both.
            child, prefix_overflowed = _exec_child_index(i + 1)
            if prefix_overflowed:
                # Wrapper chain outran the hop budget: block the chain itself.
                blocked.add(_token_basename(tokens[i + 1]))
                continue
            exec_words = [i + 1] if child in (-1, i + 1) else [i + 1, child]
            for word in exec_words:
                base = _token_basename(tokens[word])
                if _is_sed_command(base):
                    # find runs its -exec child directly, so screen a sed there too.
                    sed_indexes.append(word)
                if base in _BLOCKED_COMMANDS:
                    blocked.add(base)
                else:
                    blocked |= _blocked_matching_glob(base)

    # Backstop for blocked words at command boundaries shlex misses ($(), <(), backticks, `foo;rm`).
    lowered = command.lower()
    backstop = _BLOCKED_WORD_RE_WITH_ASSIGNMENTS if lex_raised else _BLOCKED_WORD_RE
    if backstop is not None:
        blocked.update(backstop.findall(lowered))

    # A substitution at command position synthesizes the executed word; screen the body. Variables
    # launder the same shapes, so bindings are collected first.
    if _BLOCKED_COMMANDS:
        quote_states = _shell_quote_states(command)

        def _site_expands(match: "re.Match") -> bool:
            """Whether bash expands this substitution AT command position, rather than inside an
            argument the outer command already owns."""
            opener = match.end() - 1
            state = quote_states[opener] if opener < len(quote_states) else ""
            if state in ("'", "$'", _ESCAPED_CHAR_STATE):
                return False
            if state == '"':
                # Double quotes substitute into an argument, unless the quote opens at the site.
                return match.group(0).endswith(('"$(', '"`'))
            return True

        laundered: "dict[str, str | None]" = {}
        for assign in _SUBST_ASSIGN_RE.finditer(command):
            if assign.group(1) in laundered or not _site_expands(assign):
                continue
            # An enumerating body is unknowable even when a literal (the grep selector) shows.
            opener = assign.end() - 1
            assign_body = _command_subst_body(command, opener).lower()
            if _SUBST_ENUMERATES_COMMANDS_RE.search(assign_body):
                laundered[assign.group(1)] = None
                continue
            found = sorted(_blocked_body_words(assign_body))
            laundered[assign.group(1)] = found[0] if found else None
        for printf_v in _PRINTF_V_ASSIGN_RE.finditer(command):
            laundered.setdefault(printf_v.group(1), None)
        for literal in _ASSIGN_BLOCKED_LITERAL_RE.finditer(command):
            name, value = literal.group(1), literal.group(2).strip("\"'")
            first = value.split()[0] if value.split() else ""
            base = os.path.basename(first)
            stem, ext = os.path.splitext(base)
            if ext.lower() in {".exe", ".com", ".bat", ".cmd"}:
                base = stem
            if base.lower() in _BLOCKED_COMMANDS:
                laundered.setdefault(name, base.lower())
        laundered_refs = "|".join(sorted({re.escape(n) for n in laundered}, key = len, reverse = True))
        laundered_in_body_pattern = (
            rf"\$(?:{laundered_refs}|\{{(?:{laundered_refs})\}})" if laundered_refs else None
        )
        sites = [
            site
            for site in list(_SUBST_AT_CMD_SITE_RE.finditer(command))
            + list(_SUBST_EXEC_DIRECTIVE_RE.finditer(command))
            + _case_arm_sites(command, quote_states)
            if _site_expands(site)
        ]
        if len(sites) > _MAX_SUBST_SITES:
            # Each site costs a walk to end of line (`";$(" * 10000` took 30s): refuse, fail closed.
            blocked.add(_BLOCKED_SYNTHESIZED_COMMAND)
            sites = []
        for site in sites:
            opener = site.end() - 1
            body = _command_subst_body(command, opener)
            lowered_body = body.lower()
            blocked.update(_blocked_body_words(lowered_body))
            if _subst_is_word_fragment(command, opener):
                # `"$(printf r)"m` concatenates into `rm`; only its unknowability is reportable.
                blocked.add(_BLOCKED_SYNTHESIZED_COMMAND)
            elif _SUBST_ENUMERATES_COMMANDS_RE.search(lowered_body):
                blocked.add(_BLOCKED_SYNTHESIZED_COMMAND)
            elif laundered_in_body_pattern is not None and re.search(
                laundered_in_body_pattern, body
            ):
                blocked.add(_BLOCKED_SYNTHESIZED_COMMAND)
        if laundered_refs:
            var_word = (
                r"(?P<var>\"?\$(?:(?P<bare>"
                + laundered_refs
                + r")|\{(?P<braced>"
                + laundered_refs
                + r")\})"
                # A closing quote ends the word (`"$c"`); trailing word chars join it (`${x}m` runs rm).
                r"\"?[\w.\-]*)(?=\s|$|[;&|)\]}])"
            )
            var_hits = list(
                re.finditer(_SUBST_CMD_SEP + r"\s*" + _SUBST_WRAPPER_RUN + var_word, command)
            )
            var_hits += _case_arm_sites(
                command, quote_states, re.compile(r"\)\s*" + _SUBST_WRAPPER_RUN + var_word)
            )
            for hit in var_hits:
                if _is_wrapper_flag_operand(command, hit.start("var")):
                    continue
                name = hit.group("bare") or hit.group("braced")
                precise = laundered.get(name)
                # A precise literal stands only when the expansion is the whole word.
                if precise is not None and hit.group("var").rstrip('"').endswith(
                    ("$" + name, "{" + name + "}")
                ):
                    blocked.add(precise)
                else:
                    blocked.add(_BLOCKED_SYNTHESIZED_COMMAND)

    # On a -c or /c flag, find the shell name behind it and scan the nested command.
    _SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "csh", "tcsh", "fish"}
    _SHELLS_WIN = {"cmd", "cmd.exe"}
    for i, token in enumerate(tokens):
        tok_lower = token.lower()
        is_unix_c = tok_lower == "-c" or (
            tok_lower.startswith("-") and tok_lower.endswith("c") and not tok_lower.startswith("--")
        )
        # Git Bash mangles /c, so models write //c; /k runs the payload too.
        is_win_c = _win_switch(tok_lower) in ("/c", "/k")
        if not (is_unix_c or is_win_c) or i < 1 or i + 1 >= len(tokens):
            continue
        # Skip only whole switches (/s, /v:on), never a program path like /bin/bash.
        for j in range(i - 1, -1, -1):
            prev = tokens[j]
            if prev.startswith("-"):
                continue
            # Git Bash doubles these slashes too (`cmd //v:on //c ...`).
            if is_win_c and _CMD_SWITCH_RE.fullmatch(_win_switch(prev)):
                continue
            prev_base = os.path.basename(prev).lower()
            if is_unix_c and prev_base in _SHELLS:
                blocked |= _find_blocked_commands(tokens[i + 1], posix = posix)
            elif is_win_c and prev_base in _SHELLS_WIN:
                # The cmd lexer keeps quote marks, so `"powershell` would match nothing.
                payload = tokens[i + 1]
                if len(payload) > 1 and payload[0] == '"' and payload[-1] == '"':
                    payload = payload[1:-1]
                blocked |= _find_blocked_commands(payload, posix = posix)
            break

    # `cmd /c start "" prog`: screen what start launches.
    for i, token in enumerate(tokens):
        if os.path.basename(token).lower() not in ("start", "start.exe"):
            continue
        j = i + 1
        while j < len(tokens) and _win_switch(tokens[j].lower()) in _START_SWITCHES:
            j += 2 if _win_switch(tokens[j].lower()) in _START_SWITCHES_WITH_VALUE else 1
        # A quoted first argument is the window title, so the program is the next token.
        if j < len(tokens):
            blocked |= _find_blocked_commands(tokens[j], posix = posix)
        if j + 1 < len(tokens) and _is_start_title(tokens[j]):
            k = j + 1
            while k < len(tokens) and _win_switch(tokens[k].lower()) in _START_SWITCHES:
                k += 2 if _win_switch(tokens[k].lower()) in _START_SWITCHES_WITH_VALUE else 1
            if k < len(tokens):
                blocked |= _find_blocked_commands(tokens[k], posix = posix)

    # sed `e COMMAND` runs a shell command; screen it like `bash -c`.
    sed_limit = _sed_scan_limit(len(sed_indexes))
    # Built at most once, only when a program names a variable, to stay linear.
    sed_vars: "dict[str, str] | None" = None
    sed_bindings: "list[tuple[int, str, str | None]] | None" = None
    sed_cursor = 0
    for i in sorted(set(sed_indexes)):
        # Scripts disabled by --sandbox/--posix are already excluded from the program.
        if glob_indexes is None:
            glob_indexes = (
                _unquoted_glob_indexes(command, tokens, _COMMAND_POSITION_PUNCTUATION)
                if lexed_posix
                else frozenset()
            )
        alternatives, scan_overflowed, _live = _sed_invocation(
            tokens, i, sed_limit, invocation_stops, redirect_indexes, glob_indexes
        )
        program = "\n".join(alternatives)
        if scan_overflowed:
            # Script past the scan window: block the sed rather than trust padding.
            blocked.add(_token_basename(tokens[i]))
            continue
        if _sed_program_is_a_placeholder(program):
            blocked.add(_token_basename(tokens[i]))
            continue
        if i in sed_xargs and _xargs_hides_sed_program(tokens, sed_xargs[i], i, program):
            blocked.add(_token_basename(tokens[i]))
            continue
        if "$" in program:
            # Resolve variable programs; only assignments ahead of this sed count, the last wins.
            if sed_bindings is None:
                sed_bindings = _assignment_bindings(tokens, quoted_separators)
                sed_vars = {}
            sed_cursor = _bindings_before(sed_bindings, sed_cursor, i, sed_vars)
        for alternative in alternatives:
            for variant in _sed_program_variants(alternative, sed_vars or {}):
                for payload in _sed_exec_payloads(variant):
                    if payload:
                        blocked |= _find_blocked_commands(payload, posix = posix)

    return blocked


# Holds the sandbox sitecustomize.py shim; put on the child's PYTHONPATH in _build_safe_env.
_SANDBOX_SITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_site")

# Auto mode prompting gate; fails closed: anything not provably read-only asks.

_AUTO_SAFE_TERMINAL_COMMANDS = frozenset(
    {
        "ls",
        "dir",
        "pwd",
        # cd absent: `cd /; cat etc/passwd` escapes the workdir for later relative reads.
        "cat",
        "head",
        "tail",
        # less/more absent: pager escapes can run commands or write files.
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
        # ps absent: BSD env flags dump a parent's unscrubbed env.
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
# Flags that make a read-only command write or execute.
_AUTO_UNSAFE_COMMAND_FLAGS = {
    # --files0-from reads input paths from a file, bypassing path checks.
    "sort": frozenset(
        {"-o", "--output", "--compress-program", "-T", "--temporary-directory", "--files0-from"}
    ),
    "tree": frozenset({"-o"}),
    "xxd": frozenset({"-r"}),
    # -c reads a manifest, then every path it names.
    "md5sum": frozenset({"-c", "--check"}),
    "sha1sum": frozenset({"-c", "--check"}),
    "sha256sum": frozenset({"-c", "--check"}),
    "shasum": frozenset({"-c", "--check"}),
    "cksum": frozenset({"-c", "--check"}),
    # time is a wrapper, so its flags are checked before the wrapped command.
    "time": frozenset({"-o", "--output", "-a", "--append"}),
    "rg": frozenset({"--pre", "--hostname-bin"}),
    "env": frozenset({"-C", "--chdir", "-S", "--split-string"}),
    # Process-target flags mutate another process.
    "ionice": frozenset({"-p", "-P", "-u"}),
    # `printf -v PATH %s .; ls` runs ./ls.
    "printf": frozenset({"-v"}),
    # --files0-from reads input paths from a file; find spells it -files0-from.
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
    "fd": frozenset({"-x", "--exec", "-X", "--exec-batch", "--base-directory", "--search-path"}),
    "date": frozenset({"-s", "--set"}),
    "file": frozenset({"-C", "--compile"}),
    "hostname": frozenset({"-F", "--file", "-b", "--boot"}),
}
# `hostname NAME` and `date MMDD...` mutate, so any other positional asks.
_AUTO_ARG_SENSITIVE_COMMANDS = frozenset({"hostname", "date"})
# Value-taking display flags; their value is not a clock-setting positional.
_DATE_DISPLAY_VALUE_FLAGS = frozenset({"-d", "--date", "-r", "--reference", "-f", "--file"})
# A second positional is an output file that gets overwritten.
_AUTO_SECOND_POSITIONAL_WRITES = frozenset({"uniq", "xxd"})
# Consume option values so a numeric value is not miscounted as the output positional.
_SECOND_POSITIONAL_VALUE_FLAGS = {
    "uniq": frozenset({"-f", "--skip-fields", "-s", "--skip-chars", "-w", "--check-chars"}),
    "xxd": frozenset(
        {"-c", "--cols", "-s", "--seek", "-l", "--len", "-g", "--groupsize", "-o", "--offset"}
    ),
}
# find/fd `(...)` groups reset command context, so scan every token.
_AUTO_UNSAFE_FIND_LIKE_FLAGS = _AUTO_UNSAFE_COMMAND_FLAGS["find"] | _AUTO_UNSAFE_COMMAND_FLAGS["fd"]
# Recursive readers on an absolute target read host files (grep -R TOKEN /home).
_AUTO_RECURSIVE_SEARCH = frozenset({"grep", "egrep", "fgrep", "rg", "ug", "find", "fd"})
# Always recurse; ls only with -R, gated separately.
_AUTO_RECURSIVE_LISTERS = frozenset({"tree", "du"})
# Safe wrappers that forward command position. Privilege wrappers and chroot are absent; xargs
# is absent because it appends stdin arguments this scan never sees.
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

# These read-named tools launch unsandboxed Blender on a caller-selected file.
_BLENDER_CLI_SUMMARY_TOOLS = frozenset(
    {
        "get_blendfile_summary_datablocks_for_cli",
        "get_blendfile_summary_missing_files_for_cli",
        "get_blendfile_summary_of_linked_libraries_for_cli",
        "get_blendfile_summary_path_info_for_cli",
        "get_blendfile_summary_usage_guess_for_cli",
    }
)


_AUTO_SAFE_MCP_TOOL_RE = re.compile(
    r"^(get|list|search|read|fetch|query|find|describe|show|view|lookup|"
    r"retrieve|count|status|info|help|check)(?:[_\-].*)?$",
    re.IGNORECASE,
)
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
# A credential noun anywhere asks; scoped nouns avoid primary_key/keyboard.
_AUTO_SENSITIVE_MCP_NOUN_RE = re.compile(
    r"(?:^|[_\-])(?:"
    r"secret|token|credential|password|passwd|passphrase|apikey|"
    r"(?:api|access|private|secret|signing|encryption|auth|session)[_\-]?keys?"
    r")s?(?:[_\-]|$)",
    re.IGNORECASE,
)
# runCommand -> run_Command so the term-boundary regexes match camelCase names.
_CAMEL_CASE_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# Fold other separators (catalog.get-catalog-entity) to `_`.
_MCP_TERM_SEPARATOR_RE = re.compile(r"[^A-Za-z0-9_\-]+")
# A read verb names its subject, so impact/runtime-noun patterns must not fire on it.
_AUTO_READ_MCP_VERB_RE = re.compile(
    r"(?:^|[_\-])(?:get|list|read|search|find|fetch|query|describe|show|view|"
    r"inspect|status|info|count|exists|lookup|browse|preview|download|export|"
    r"history|log|logs|diff|compare|summarize|summarise)(?:[_\-]|$)",
    re.IGNORECASE,
)
# Runtime nouns only count when nothing reads.
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
# Code/command runners run on the MCP server outside the sandbox; whole segments only.
_AUTO_EXEC_MCP_TOOL_RE = re.compile(
    r"(?:^|[_\-])(?:"
    r"exec|execute|run|eval|spawn|invoke|launch|"
    r"shell|bash|zsh|powershell|pwsh|terminal|subprocess|interpreter|"
    # A bare runtime name is an execution tool even without a verb.
    r"python[0-9.]*|node|nodejs|deno|bun|ruby|perl|php|code|script|repl|sandbox|notebook"
    r")(?:[_\-]|$)",
    re.IGNORECASE,
)
# Destructive verbs as whole segments prompt even without a mutation marker (not `undelete`).
_AUTO_DESTRUCTIVE_MCP_VERB_RE = re.compile(
    r"(?:^|[_\-])(?:"
    r"delete|destroy|drop|purge|wipe|truncate|erase|remove|unlink|"
    r"teardown|revoke|terminate|uninstall|clear|reset|empty|flush|prune|expire"
    r")(?:[_\-]|$)",
    re.IGNORECASE,
)
# Names without separators (runcommand) miss segment boundaries; match compounds directly.
_MCP_EXEC_VERBS = r"execute|exec|run|eval|spawn|invoke|launch|start"
_MCP_EXEC_OBJECTS = r"command|cmd|shell|script|code|process|program|bash|terminal|proc|task|job"
_AUTO_EXEC_MCP_COMPOUND_RE = re.compile(
    r"(?:^|[_\-])(?:"
    rf"(?:{_MCP_EXEC_VERBS})(?:{_MCP_EXEC_OBJECTS})"
    rf"|(?:{_MCP_EXEC_OBJECTS})(?:{_MCP_EXEC_VERBS})"
    r")(?:[_\-]|$)",
    re.IGNORECASE,
)
# Verbs that may run without a prompt: reads and ordinary writes.
_AUTO_KNOWN_MCP_VERBS = frozenset(
    {
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


# Verbs already gated above; `undelete` reverses one, so it stays screenable.
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
    """Whether any term of an MCP tool name is a verb this classifier knows. A name with none of
    them cannot be screened, so the caller fails closed."""
    for part in re.split(r"[_\-]+", tool_name.lower()):
        if not part:
            continue
        if part in _AUTO_KNOWN_MCP_VERBS:
            return True
        for prefix in ("un", "re"):
            if part.startswith(prefix) and part[len(prefix) :] in _AUTO_MCP_VERB_VOCAB:
                return True
    return False


# Privilege verbs match alone; soft verbs (assign/add/set) only next to a privilege noun.
_AUTO_PRIVILEGE_MCP_VERB_RE = re.compile(
    r"(?:^|[_\-])(?:grant|authorize|authorise|elevate|escalate|impersonate|sudo|promote)(?:[_\-]|$)",
    re.IGNORECASE,
)
# Money movement and other irreversible external effects.
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

# Modules whose import alone signals side effects (spawn, network, bulk fs, raw memory).
_AUTO_UNSAFE_PY_MODULES = frozenset(
    {
        "subprocess",
        "shutil",
        "socket",
        "_socket",
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
        "huggingface_hub",
        "websockets",
        "socketserver",
        "ftplib",
        "smtplib",
        "telnetlib",
        "paramiko",
        "imaplib",
        "poplib",
        "nntplib",
        "xmlrpc",
        "webbrowser",
        "tempfile",
        "pickle",
        "marshal",
        "shelve",
        "dill",
        "dbm",
        "sqlite3",
        "runpy",
        "ensurepip",
        "venv",
    }
)
# Fs-mutating / spawning attributes regardless of how the module was bound.
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
        "symlink_to",
        "hardlink_to",
        "link_to",
        "mkfifo",
        "mknod",
        "utime",
        "setxattr",
        "removexattr",
        "import_module",
        # extract/extractall allow zip-slip even for a single member.
        "exec_module",
        "extractall",
        "extract",
        "FileIO",
        "create_subprocess_exec",
        "create_subprocess_shell",
        "subprocess_exec",
        "subprocess_shell",
        "open_connection",
        "create_connection",
        "create_server",
        "create_unix_connection",
        "create_unix_server",
        "start_server",
        "start_unix_server",
        "open_unix_connection",
        "create_datagram_endpoint",
        "sock_connect",
        "chdir",
        "fchdir",
        "run_path",
        "run_module",
        "FunctionType",
        "read_pickle",
    }
)
# Gated by receiver module since bare `load` is too common.
_AUTO_UNSAFE_PY_LOAD_MODULES = frozenset({"torch", "joblib", "cloudpickle", "yaml"})
# numpy loaders that unpickle under allow_pickle -> its positional index. The gate checks positions 1 and 2 of all of
# them, so an alias pointing at another loader cannot shift the flag out of view.
_NUMPY_PICKLE_FLAG_POS = {"load": 2, "read_array": 1, "NpzFile": 2}
# The load entry points on those modules. yaml.load runs whatever its Loader= builds, and !!python/object/apply in the
# data is a call, so it asks like the pickle-backed ones. yaml.safe_load is untouched.
_AUTO_UNSAFE_PY_LOAD_ATTRS = frozenset({"load", "load_all"})
# Ordinary words, so matched by receiver.
_AUTO_UNSAFE_PY_LOAD_CLASSES = frozenset({"Loader", "Constructor"})
# Unique names, matched anywhere rather than by tracing bindings.
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
# Writers that persist without open(); method calls only.
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
        # to_string omitted: overwhelmingly display-only.
        "to_html",
        "to_markdown",
        "to_latex",
        "to_clipboard",
        "to_gbq",
        "imwrite",
        "imsave",
        "write_image",
        "write_html",
        "save_pretrained",
        "save_file",
        "save_model",
        "save_weights",
        "save_lora",
        "save_checkpoint",
        # Opens a log file for write on construction.
        "FileHandler",
        "WatchedFileHandler",
        "RotatingFileHandler",
        "TimedRotatingFileHandler",
        "memmap",
        "open_memmap",
        "ExcelWriter",
        "HDFStore",
        "writedoc",
    }
)
# Mode is the 2nd arg like open, so gated only in write mode.
_ARCHIVE_CTOR_NAMES = frozenset({"ZipFile", "TarFile", "GzipFile", "BZ2File", "LZMAFile"})
_ARCHIVE_CTOR_MODULES = {
    "zipfile": "ZipFile",
    "tarfile": "TarFile",
    "gzip": "GzipFile",
    "bz2": "BZ2File",
    "lzma": "LZMAFile",
}
# Top-level open() takes mode second, so `from gzip import open as gopen` is gated.
_OPEN_ALIAS_MODULES = frozenset({"gzip", "bz2", "lzma"})
# Call their first argument per item, so a writer alias passed in runs.
_HIGHER_ORDER_INVOKERS = frozenset({"map", "filter", "starmap", "reduce"})
_PY_WRITE_MODE_RE = re.compile(r"[wax+]")
# Tells Path.open("w") from ZipFile.open("name.txt").
_PY_MODE_LITERAL_RE = re.compile(r"^[rwxa][btru+]*$")
# `remove` is gated on `os` only so list.remove() stays out.
_PY_DESTRUCTIVE_FS_ATTRS = frozenset({"unlink", "rmtree", "rmdir", "removedirs"})
_PY_PROCESS_KILL_ATTRS = frozenset({"kill", "terminate", "send_signal", "suspend"})
_PY_PROCESS_MODULES = frozenset({"psutil"})
# Gated only on the os module so same-named methods elsewhere stay out.
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
_PY_DESTRUCTIVE_FS_MODULES = ("os", "posix", "nt", "shutil", "pathlib")


# Credential names are stored in pieces so this file does not carry them verbatim; split new
# ones the same way.
def _joined(*parts) -> str:
    """Concatenate parts; a tuple part is one name split into pieces."""
    return "".join("".join(part) for part in parts)


# Credential paths, plus `..` traversal out of the session workdir.
_SENSITIVE_PATH_RE = re.compile(
    r"(?:^|[/\\])\.(?:ssh|aws|azure|gnupg|docker|kube|config/gcloud|config/gh)(?:[/\\]|$)"
    + _joined((r"|\.(?:net", r"rc|npmrc|pypirc|git-cred", r"entials|env)(?:$|[/\\.\s'\"])"))
    # Shell startup files and autostart dirs: a write persists to the next login.
    + r"|(?:^|[/\\\s'\"=])\.(?:bashrc|bash_profile|bash_login|bash_logout|bash_aliases"
    r"|profile|zshrc|zprofile|zshenv|zlogin|zlogout|kshrc|cshrc|tcshrc|login"
    r"|xprofile|xinitrc|xsession)(?:$|[/\\\s'\"])"
    r"|(?:^|[/\\])\.config[/\\](?:autostart|systemd[/\\]user|environment\.d)(?:[/\\]|$)"
    + _joined((r"|id_r", r"sa"), (r"|id_ed", r"25519"), (r"|id_ec", r"dsa"), (r"|id_d", r"sa"))
    # Only the HF credential files; the rest of the cache is model data.
    + r"|(?:^|[/\\])\.?huggingface[/\\](?:token|stored_tokens)(?:$|[/\\.\s'\"])"
    # /etc/ssh holds host keys; the trailing group is system persistence hooks.
    + _joined((r"|cred", r"entials"), (r"|/etc/(?:pas", r"swd|sh", r"adow|sudoers|ssh(?:[/\\]|$)"))
    + r"|cron[^/\\]*(?:[/\\]|$)|profile\.d(?:[/\\]|$)|systemd(?:[/\\]|$)"
    r"|ld\.so\.preload(?:$|[/\\.\s'\"])|ld\.so\.conf|rc\.local|init\.d(?:[/\\]|$))"
    # bash opens /dev/tcp and /dev/udp as network sockets.
    r"|/dev/(?:tcp|udp)/"
    r"|/(?:var/)?run/secrets(?:[/\\]|$)"
    # procfs leaks process env/args/memory; fd/ links to open files.
    r"|/proc/[^/\s'\"]+/(?:task/[^/\s'\"]+/)?(?:environ|cmdline|mem|maps|fd)\b"
    # A .pem/.key file (basename before the extension), not a bare ".key" (e.g. a jq '.key' filter).
    r"|\w[\w.-]*\.(?:pem|key)(?:$|[\s'\"])",
    re.IGNORECASE,
)
# $STUDIO_HOME/auth holds credentials in the clear and tools run as the same user. A name ends
# where the shell ends a word, including command punctuation.
_WORD_END = r"(?:$|[\s'\";&|)(<>`])"
_STUDIO_CREDENTIAL_BASENAME_RE = re.compile(
    r"(?:^|[/\\\s'\"=])(?:\.cli_api_key_[^/\\\s'\";&|)(<>`]*|\.bootstrap_password|\.desktop_secret)"
    + _WORD_END
    # The per-launch key file, bare or as a glob (find / -name 'llama_api_key_*').
    + r"|(?:^|[/\\\s'\"=])llama_api_key_[^/\\\s'\";&|)(<>`]*"
    + _WORD_END
    # Path form only: the bare name is an ordinary identifier. auth.db is absent for the same reason,
    # and Studio's copy is covered by the auth-directory patterns below.
    + r"|[/\\]llama_api_key(?:_\w+)?"
    + _WORD_END
    # `unsloth start` keeps coding-agent keys here; path form only.
    + r"|[/\\]agent_api_key\.json"
    + _WORD_END,
    re.IGNORECASE,
)
_STUDIO_AUTH_DIR_RE = re.compile(
    r"(?:^|[/\\\s'\"=])\.unsloth[/\\]studio[/\\]auth(?:[/\\]|" + _WORD_END + r")",
    re.IGNORECASE,
)

# Only with a `cd` into the studio root, never a mere mention.
_BARE_AUTH_SEGMENT_RE = re.compile(
    r"(?:^|[/\\\s'\"=])auth(?:[/\\]|" + _WORD_END + r")", re.IGNORECASE
)

# Prefilter: a new pattern needs a hint here or it never runs.
# normcase is identity on POSIX except macOS.
_CASE_SENSITIVE_PATHS = os.path.normcase("A") == "A" and sys.platform != "darwin"


_STUDIO_CREDENTIAL_HINTS = (
    "auth",
    ".cli_api_key",
    ".bootstrap_password",
    ".desktop_secret",
    "llama_api_key",
    "agent_api_key",
)

# Both reach the tool subprocess env, so `$STUDIO_HOME/auth` is the real path.
_STUDIO_HOME_ENV_VARS = ("UNSLOTH_STUDIO_HOME", "STUDIO_HOME")


def _same_directory(left: str, right: str) -> bool:
    """Whether two spellings name the same directory.

    normcase because Windows paths are case-insensitive, and realpath because `studio_root()`
    resolves aliases: a `STUDIO_HOME` that is a symlink or junction to the configured root is the
    root, and reading it as somewhere else drops every spelling of the variable from the guard.
    """

    def tidy(path: str) -> str:
        return os.path.normcase(os.path.normpath(path))

    if tidy(left) == tidy(right):
        return True
    try:
        return tidy(os.path.realpath(left)) == tidy(os.path.realpath(right))
    except OSError:
        return False


def _studio_home_variable_spellings(resolved: "str | None" = None) -> "list[str]":
    """`$VAR`, `${VAR}`, `%VAR%` and `$env:VAR` for each studio-home variable (sh, cmd, PowerShell).

    A variable that is SET to some other directory is skipped. `STUDIO_HOME` is a generic name, and
    another application can own it; with the spelling registered unconditionally, a command naming
    that application's directory was refused in every permission mode even though it never came near
    this install. A variable that is unset stays registered: the child cannot expand it either, so
    the spelling reaches nothing, and dropping it would only widen the guard for no gain.
    """
    out: "list[str]" = []
    for var in _STUDIO_HOME_ENV_VARS:
        value = os.environ.get(var)
        if value and resolved:
            try:
                points_here = _same_directory(os.path.expanduser(value), resolved)
            except Exception:  # noqa: BLE001 - an unreadable value must not break classification
                points_here = True
            if not points_here:
                continue
        out.extend((f"${var}", f"${{{var}}}", f"%{var}%", f"$env:{var}"))
    return out


_studio_auth_markers_cache: "tuple | None" = None


def _studio_auth_dir_markers() -> tuple:
    """``(plain spellings, variable spellings, "cd into the studio root" pattern)`` for this install.

    Each spelling is a ``(marker, canonical marker)`` pair, both lowercased, so the per-call path
    does no string building. The variable spellings (``$HOME/...``, ``$STUDIO_HOME/auth``, ``~/...``)
    are kept apart because a text containing no ``$``, ``%`` or ``~`` cannot match one, and the
    terminal classifier runs this once per candidate token of every command.

    Resolved once per process: STUDIO_HOME is fixed at startup, and a custom UNSLOTH_STUDIO_HOME is
    only covered by asking for the real root rather than assuming the default layout. A failure is
    not cached, so a root that could not be resolved during startup import ordering does not leave
    the guard half-blind for the process lifetime."""
    global _studio_auth_markers_cache
    if _studio_auth_markers_cache is not None:
        return _studio_auth_markers_cache
    try:
        from utils.paths.storage_roots import auth_root
        resolved = str(auth_root())
    except Exception:  # noqa: BLE001 - an unresolvable root leaves the literal patterns above
        return (), (), None
    if not resolved:
        return (), (), None
    auth_markers = [resolved]
    variable_markers: "list[str]" = []
    root_markers = [os.path.dirname(resolved.rstrip("/\\"))]
    home = os.path.expanduser("~")
    # Boundary-aware: a plain startswith reads /home/u2 as under /home/u and mints a "~2/..." marker.
    if home and (resolved == home or resolved.startswith(home.rstrip(os.sep) + os.sep)):
        for target, base in ((variable_markers, resolved), (root_markers, root_markers[0])):
            tail = base[len(home.rstrip(os.sep)) :]
            target.extend(("~" + tail, "$HOME" + tail, "${HOME}" + tail))
    for spelling in _studio_home_variable_spellings(os.path.dirname(resolved.rstrip("/\\"))):
        root_markers.append(spelling)
        variable_markers.extend((spelling + "/auth", spelling + "\\auth"))
    roots = [m for m in root_markers if m and m not in ("/", "\\")]
    # The root must end where it matched or continue into `auth`, and only at a command position.
    cd_re = (
        re.compile(
            r"(?:^|[;&|({\n]\s*|\b(?:then|do|else|if|elif|while|until)\s+)(?:(?:builtin|command|exec)\s+)*"
            r"(?:cd|pushd)\s+(?:/d\s+)?[\"']?(?:"
            + "|".join(re.escape(m) for m in roots)
            + r")(?:[\"']|\s|[;&|]|$|[/\\]auth(?![\w-]))",
            re.IGNORECASE,
        )
        if roots
        else None
    )

    def pairs(names: "list[str]") -> tuple:
        out = []
        for name in names:
            if not name:
                continue
            lowered = name.lower()
            # Keep the original spelling: `Auth` differs from `auth` on case-sensitive filesystems.
            out.append((lowered, _canonical_path_text(lowered), name))
        return tuple(out)

    _studio_auth_markers_cache = (pairs(auth_markers), pairs(variable_markers), cd_re)
    return _studio_auth_markers_cache


# Names neither the path nor the value: this string goes back to the model.
_STUDIO_CREDENTIAL_BLOCKED = (
    "Blocked for safety: Unsloth Studio's authentication directory holds this install's own "
    "credentials and is not readable by tools."
)


def _canonical_path_text(text: str) -> str:
    """Rewrite *text* into the one spelling the OS would resolve it to, lexically.

    Separators are unified to "/", then `//` and `/./` collapse and `x/..` pairs cancel. The OS
    opens `<home>/bin/../auth/auth.db` and `C:\\Studio\\.\\auth\\auth.db` as the protected database,
    so without this the guard matched only the tidiest spelling of a path and every equivalent one
    walked past it.

    Lexical on purpose: no `realpath`, so no filesystem access and nothing to race. That makes it
    wrong for a path whose parent is a symlink, which is the accepted limit here -- this is an extra
    candidate spelling, never a replacement, so a match found in the raw text still counts.
    """
    unified = text.replace("\\", "/")
    out: "list[str]" = []
    for index, segment in enumerate(unified.split("/")):
        # Collapse `//`; the first two positions keep a leading "/" and UNC prefix.
        if segment == "" and index > 1:
            continue
        if segment == ".":
            continue
        if segment == ".." and out and out[-1] not in ("", ".."):
            out.pop()
            continue
        out.append(segment)
    return "/".join(out)


_GLOB_META_RE = re.compile(r"[*?\[]")
_BRACKET_CLASS_RE = re.compile(r"\[[^\]/\s]{1,64}\]")
# A one-char class is that char: `[a][u][t][h]` IS `auth`.
_SINGLETON_CLASS_RE = re.compile(r"\[([^\]/\s!^-])\]")


# The shell drops a backslash before a word char (`c\d` is `cd`); separators left alone.
_ESCAPED_WORD_CHAR_RE = re.compile(r"\\(\w)")
# The bare `~` only counts at the head of a path.
_HOME_VARIABLE_RE = re.compile(r"\$\{HOME\}|\$HOME\b|%HOME%|(?<![\w~.])~(?=[/\\])", re.IGNORECASE)
# Bypass sets PWD to the tool workdir, so `$PWD/../..` walks up from the sandbox.
_CWD_VARIABLE_RE = re.compile(r"\$\{PWD\}|\$PWD\b|%CD%|\$env:PWD\b", re.IGNORECASE)


def _glob_can_name_the_marker(lowered: str, marker: str) -> bool:
    """True when *lowered*, read as a shell glob, can expand to *marker* or something under it.

    `sqlite3 ../../a?th/auth.db` never spells the auth directory, but bash expands `a?th` to `auth`
    before sqlite3 opens anything, so comparing the literal text alone let the database through.
    Matched segment by segment, because a shell wildcard does not cross a separator while
    `fnmatch`'s does.

    A segment whose literal characters are all metacharacters is skipped: `ls <studio root>/*` names
    the auth directory only in the sense that listing a parent does, and refusing it would break
    ordinary work for no secret read.
    """
    candidate = lowered.split("/")
    wanted = marker.split("/")
    if len(candidate) < len(wanted):
        return False
    aligned = candidate[len(wanted) - 1]
    if not _GLOB_META_RE.sub("", aligned).strip("]-"):
        return False
    return all(
        fnmatch.fnmatchcase(want, have) for want, have in zip(wanted, candidate[: len(wanted)])
    )


# A command naming the studio root and one of these is enumerating for it.
_CREDENTIAL_BASENAME_RE = re.compile(
    r"(?<![\w.-])(?:auth\.db|\.desktop_secret|\.bootstrap_password|\.cli_api_key[\w.-]*"
    r"|llama_api_key[\w.-]*|agent_api_key[\w.-]*)(?![\w-])",
    re.IGNORECASE,
)


_STUDIO_HOME_ASSIGN_RE = re.compile(
    r"(?:^|[;&|(\s])(?:export\s+|set\s+)?(" + "|".join(_STUDIO_HOME_ENV_VARS) + r")\s*=",
    re.IGNORECASE,
)


def _assignment_is_a_command_prefix(text: str, value_start: int) -> bool:
    """True when a command follows the assignment's value on the same simple command.

    `H=/tmp cat x` is a prefix assignment; `H=/tmp; cat x` is an assignment that stands alone. The
    value ends at the first unquoted separator, so the quotes are tracked while scanning it.
    """
    index = value_start
    quote = ""
    while index < len(text):
        char = text[index]
        if quote:
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char in " \t;&|\n":
            break
        index += 1
    while index < len(text) and text[index] in " \t":
        index += 1
    return index < len(text) and text[index] not in ";&|\n"


def _assignment_is_inert(text: str, index: int) -> bool:
    """Whether an assignment at *index* binds nothing for the commands that follow it.

    Inside quotes it is DATA that the command merely prints, and inside `( ... )` or
    `$( ... )` it binds only the SUBSHELL, so the outer commands still expand the
    inherited value."""
    quote = ""
    escaped = False
    depth = 0
    for character in text[:index]:
        if escaped:
            escaped = False
        elif character == "\\" and quote != "'":
            # A backslash escapes the next char wherever a single quote is not open.
            escaped = True
        elif quote:
            if character == quote:
                quote = ""
        elif character in "'\"":
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth = max(depth - 1, 0)
    return bool(quote) or depth > 0


def _assignment_inert_states(text: str) -> "list[bool]":
    """`_assignment_is_inert(text, i)` for every i in one pass (index len(text) included)."""
    states = []
    quote = ""
    escaped = False
    depth = 0
    for character in text:
        states.append(bool(quote) or depth > 0)
        if escaped:
            escaped = False
        elif character == "\\" and quote != "'":
            escaped = True
        elif quote:
            if character == quote:
                quote = ""
        elif character in "'\"":
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth = max(depth - 1, 0)
    states.append(bool(quote) or depth > 0)
    return states


def _rebinds_the_studio_home_first(text: str) -> bool:
    """True when *text* assigns a studio home variable BEFORE any use of it.

    The shell assignment is what the child expands, whether or not the backend sets the same name,
    so `UNSLOTH_STUDIO_HOME=/tmp/project; cat "$UNSLOTH_STUDIO_HOME/auth/config.json"` reads a
    project file and not this install's directory. An assignment that comes AFTER a use rebinds
    nothing for that use, and expanding it would hide a real read of the install's own auth
    directory, so the ordering is what decides it.
    """
    lowered = text.lower()
    assignments: "dict[str, int]" = {}
    for match in _STUDIO_HOME_ASSIGN_RE.finditer(text):
        # A printed or subshell-only assignment does not rebind the later expansion.
        if _assignment_is_inert(text, match.end()):
            continue
        # A prefix assignment does not govern its own command's argument expansion; only a
        # standalone assignment rebinds what follows.
        if (
            _assignment_is_a_command_prefix(text, match.end())
            and (os.environ.get(match.group(1).upper()) or "").strip()
        ):
            # Only when set: an unset variable expands to nothing.
            continue
        # The last assignment of a name is the one the expansion applies.
        assignments[match.group(1).lower()] = match.end()
    # Every assigned name must be safe: one name's assignment must not authorize rewriting another.
    rebinds = False
    for name, position in assignments.items():
        uses = [
            found
            for found in (
                lowered.find(f"${name}"),
                lowered.find(f"${{{name}}}"),
                lowered.find(f"%{name}%"),
            )
            if found != -1
        ]
        if uses and min(uses) < position:
            return False
        rebinds = True
    return rebinds


_studio_root_spellings_cache: "tuple | None" = None


def _studio_root_spellings() -> "list[str]":
    """Every lowered spelling of the Studio root: the literal path and its environment variables.

    The variables come from `_studio_home_variable_spellings`, which drops one that is SET to some
    other directory: `STUDIO_HOME` is a generic name another application can own, and registering it
    unconditionally refused `find "$STUDIO_HOME" ...` against that application's tree.
    """
    global _studio_root_spellings_cache
    markers = _studio_auth_dir_markers()[0]
    if _studio_root_spellings_cache is not None and _studio_root_spellings_cache[0] is markers:
        return _studio_root_spellings_cache[1]
    root = _studio_home_for_guard()
    if not root:
        return []
    # Both separator styles: a Windows root reaches source as `c:\\dir` and `c:/dir`.
    folded = _folded_word(root)
    spellings = [root.lower()]
    if os.sep == "\\":
        spellings += [folded, folded.replace("/", "\\")]
    spellings.extend(spelling.lower() for spelling in _studio_home_variable_spellings(root))
    spellings = list(dict.fromkeys(spellings))
    _studio_root_spellings_cache = (markers, spellings)
    return spellings


def _text_names_the_studio_root(text: str) -> bool:
    """True when *text* names the Studio root directory, literally or by one of its variables."""
    lowered = text.lower()
    spellings = _studio_root_spellings()
    if any(spelling in lowered for spelling in spellings):
        return True
    # Escaped spellings need a backslash to exist, so only rewrite then.
    if "\\" not in lowered:
        return False
    for unescaped in (lowered.replace("\\ ", " "), lowered.replace("\\\\", "\\")):
        if unescaped != lowered and any(spelling in unescaped for spelling in spellings):
            return True
    return False


def _quoted_words(text: str) -> "list[str]":
    """Split on whitespace, honouring quotes but NOT backslash escapes.

    `shlex.split(posix = True)` eats the separators of a Windows path, so
    `find "C:\\Users\\me\\Unsloth Studio" ...` came back as one mangled word and matched no root.
    """
    lexer = shlex.shlex(text, posix = True)
    lexer.whitespace_split = True
    lexer.escape = ""
    try:
        return list(lexer)
    except ValueError:
        return text.split()


def _folded_word(word: str) -> str:
    """A word reduced to the directory it names: `"<root>"/.` and `<root>//` are both `<root>`."""
    # The backslash escapes a space here; remove it before normalising separators.
    folded = word.lower().replace("\\ ", " ").replace("\\", "/")
    while folded.endswith(("/.", "/")):
        folded = folded[:-2] if folded.endswith("/.") else folded[:-1]
    return folded


def _names_the_studio_root_itself(text: str) -> bool:
    """True when one WORD of the command is the root itself, so the walk starts there.

    Word by word, because the spelling appearing anywhere is not enough: the pattern in
    `grep -rn 'cd <root>' src/` searches a project, and `<root>/projects/p/sandbox` is ordinary work
    inside a project. Only a bare `<root>` (with any trailing separator) is the directory whose walk
    reaches `auth/`.
    """
    # Folded the same way as the words, or a Windows root never matches.
    spellings = [_folded_word(spelling) for spelling in _studio_root_spellings()]
    if not spellings:
        return False
    words = _quoted_words(text)
    if "\\ " in text:
        # Lexer escapes are off for Windows paths, which splits `Studio\\ Home`; re-join it.
        words = words + [
            word.replace("\x00", " ") for word in _quoted_words(text.replace("\\ ", "\x00"))
        ]
    return any(_folded_word(word) in spellings for word in words)


# Tree walkers that emit or copy contents; plain listings name files without reading them.
# `7z a` recurses with no flag.
_STUDIO_WALK_COMMANDS = frozenset({"rsync", "tar", "cpio", "rg", "ag", "ack", "7z", "7za", "7zr"})
_STUDIO_WALK_FLAG_COMMANDS = frozenset({"grep", "egrep", "fgrep", "cp", "scp", "zip"})
_STUDIO_FIND_ACTIONS = frozenset({"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint"})
_STUDIO_WALK_SPLIT_RE = re.compile(r"[\s;&()<>]+")


def _command_words(text: str) -> "list[str]":
    """The word in COMMAND position of each pipeline segment, lowered and basenamed.

    `echo tar "$STUDIO_HOME"` runs `echo`; `tar` is data it prints. Reading every non-option token
    as an executable refused that. Assignments and the usual wrappers are stepped over so
    `env -u X tar ...` still reports `tar`.
    """
    words: "list[str]" = []
    for segment in _COMMAND_SEPARATOR_RE.split(text.lower()):
        skip_next = False
        for token in _quoted_words(segment):
            if skip_next:
                skip_next = False
                continue
            if token.startswith("-"):
                skip_next = token in _WRAPPER_VALUE_OPTIONS
                continue
            if "=" in token:
                continue
            base = os.path.basename(token.strip("\"'"))
            if base in _WALK_TRANSPARENT_WRAPPERS or _WRAPPER_DURATION_RE.match(base):
                continue
            words.append(base)
            break
    return words


_COMMAND_SEPARATOR_RE = re.compile(r"[;&|()\n]+|&&|\|\|")
_STUDIO_WALK_NAME_HINTS = frozenset(_STUDIO_WALK_COMMANDS | _STUDIO_WALK_FLAG_COMMANDS | {"find"})
_WRAPPER_VALUE_OPTIONS = frozenset(
    {"-u", "--unset", "-n", "-c", "-i", "-p", "-C", "--chdir", "-k", "--kill-after", "-s"}
)
_WRAPPER_DURATION_RE = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")
_WALK_TRANSPARENT_WRAPPERS = frozenset(
    {
        "env",
        "nice",
        "ionice",
        "nohup",
        "time",
        "timeout",
        "sudo",
        "command",
        "builtin",
        "exec",
        "stdbuf",
        "xargs",
        "busybox",
    }
)


def _walks_a_tree_reading_it(text: str) -> bool:
    """True when the command recursively reads or copies a whole directory tree."""
    lowered = text.lower()
    # Skip the lexer pass when no walker is named at all.
    if not any(name in lowered for name in _STUDIO_WALK_NAME_HINTS):
        return False
    names = set(_command_words(text))
    if not names:
        return False
    if names & _STUDIO_WALK_COMMANDS:
        return True
    tokens = [token for token in _STUDIO_WALK_SPLIT_RE.split(text.lower()) if token]
    if "find" in names and (any(token in _STUDIO_FIND_ACTIONS for token in tokens) or "|" in text):
        return True
    if names & _STUDIO_WALK_FLAG_COMMANDS:
        for token in tokens:
            if token in ("--recursive", "--archive"):
                return True
            if token.startswith("-") and not token.startswith("--") and set(token[1:]) & {"r", "a"}:
                return True
    return False


_ABSOLUTE_MARKER_RE = re.compile(r"^(?:[/\\]|[a-z]:[/\\])")


def _marker_is_absolute(marker: str) -> bool:
    return bool(_ABSOLUTE_MARKER_RE.match(marker))


def _marker_is_a_path_segment(lowered: str, marker: str) -> bool:
    """True when ``marker`` appears in ``lowered`` as a whole path, not merely as a prefix.

    ``<home>/auth`` must match ``<home>/auth/auth.db`` and a bare ``<home>/auth``, but not
    ``<home>/authors/notes.txt`` or ``<home>/auth-backup``.
    """
    start = lowered.find(marker)
    while start != -1:
        end = start + len(marker)
        ends_here = end == len(lowered) or lowered[end] in "/\\'\" \t\r\n;:&|)"
        # Leading boundary too: `/mnt/backup<home>/auth` is not the configured path.
        begins_here = (
            start == 0
            or not _marker_is_absolute(marker)
            or lowered[start - 1] in "'\" \t\r\n;&|(=,"
        )
        if ends_here and begins_here:
            return True
        start = lowered.find(marker, start + 1)
    return False


def _glob_text_can_name_the_auth_dir(lowered: str) -> bool:
    """True when *lowered*, read as a shell glob, can expand onto the auth directory or below it."""
    auth_markers, variable_markers, _cd_re = _studio_auth_dir_markers()
    markers = auth_markers + (variable_markers if ("$" in lowered or "%" in lowered) else ())
    if not markers:
        return False
    candidates = {lowered, _canonical_path_text(lowered)}
    return any(
        _glob_can_name_the_marker(candidate, canonical_marker)
        for _marker, canonical_marker, _original in markers
        for candidate in candidates
    )


def _references_studio_credential(text: str) -> bool:
    """True if *text* names Studio's auth directory or one of the credential files in it."""
    if not text:
        return False
    lowered = text.lower()
    if not any(hint in lowered for hint in _STUDIO_CREDENTIAL_HINTS):
        # A wildcard can name the directory without any hint (`../../a?th/.b*`).
        if not _GLOB_META_RE.search(lowered):
            return False
        return _glob_text_can_name_the_auth_dir(lowered)
    normalized = _REDUNDANT_SLASH_RE.sub("", text)
    lowered_normalized = normalized.lower()
    unescaped = text.replace("\\ ", " ") if "\\ " in text else text
    # The shell concatenates fragments, so drop quotes before matching.
    if '"' in unescaped or "'" in unescaped:
        unescaped = unescaped.replace('"', "").replace("'", "")
    canonical = _canonical_path_text(text)
    lowered_canonical = canonical.lower()
    canonical_candidates = {canonical}
    if unescaped is not text:
        canonical_candidates.add(_canonical_path_text(unescaped))
    if any(
        pattern.search(candidate)
        for pattern in (_STUDIO_CREDENTIAL_BASENAME_RE, _STUDIO_AUTH_DIR_RE)
        for candidate in ({text, normalized} | canonical_candidates)
    ):
        return True
    auth_markers, variable_markers, cd_into_root_re = _studio_auth_dir_markers()
    if variable_markers and ("$" in lowered or "%" in lowered or "~" in lowered):
        auth_markers = auth_markers + variable_markers
    if not auth_markers:
        return False
    # Segment-bounded so `<home>/authors` is not refused.
    lowered_raw = {lowered, lowered_normalized}
    lowered_canonicals = {lowered_canonical} | {c.lower() for c in canonical_candidates}
    if unescaped is not text:
        lowered_raw.add(unescaped.lower())
    matched = [
        original
        for marker, canonical_marker, original in auth_markers
        if any(_marker_is_a_path_segment(candidate, marker) for candidate in lowered_raw)
        or any(
            _marker_is_a_path_segment(candidate, canonical_marker)
            for candidate in lowered_canonicals
        )
    ]
    if matched:
        # On case-sensitive filesystems require the original case for literal markers; variable and
        # `~` spellings keep the folded answer.
        raw_candidates = {text, normalized, unescaped, canonical}
        if _CASE_SENSITIVE_PATHS and not any(
            original in candidate
            for original in matched
            if original[:1] not in ("$", "~", "%")
            for candidate in raw_candidates
        ):
            literal = [o for o in matched if o[:1] not in ("$", "~", "%")]
            if literal and len(literal) == len(matched):
                return False
        return True
    # Glob check per path-shaped token so it lines up with an absolute marker.
    if _GLOB_META_RE.search(lowered):
        glob_tokens = {t for c in lowered_canonicals for t in _PATH_TOKEN_RE.findall(c)}
        glob_tokens.update(lowered_canonicals)
        if any(
            _glob_can_name_the_marker(token, canonical_marker)
            for _marker, canonical_marker, _original in auth_markers
            for token in glob_tokens
            if _GLOB_META_RE.search(token)
        ):
            return True
    return bool(
        cd_into_root_re is not None
        and _BARE_AUTH_SEGMENT_RE.search(text)
        and any(cd_into_root_re.search(candidate) for candidate in (text, unescaped))
    )


_PATH_TOKEN_RE = re.compile(r"[^\s'\"()\[\]{},;|&<>]+")
_TRAVERSAL_TOKEN_RE = re.compile(r"[^\s'\"()\[\]{},;|&<>]*\.\.[^\s'\"()\[\]{},;|&<>]*")
_RELATIVE_PATH_TOKEN_RE = re.compile(r"[^\s'\"()\[\]{},;|&<>]*[/\\][^\s'\"()\[\]{},;|&<>]*")
# Case-insensitive: the shell is `cmd /c` on Windows without a trusted bash.
_CD_TARGET_RE = re.compile(
    # `builtin cd`, `command cd`, `! cd` and `time cd` move like a bare `cd`.
    r"(?:^\s*|[;&|({\n]\s*|\b(?:then|do|else|if|elif|while|until)\s+)"
    r"(?:(?:builtin|command|exec|nohup)\s+|time\s+(?:-p\s+)?|!\s*|[A-Za-z_]\w*=[^\s;&|()]*\s+)*"
    r"(?:cd|pushd)\s+"
    r"(?:(?:-[LPe@]+|--|/d)\s+)*([^\s;&|)]+)",
    re.IGNORECASE,
)
# `cd -` and `popd` return; matched in one pass so order is kept.
_DIRECTORY_MOVE_RE = re.compile(
    r"(?:^\s*|[;&|({\n]\s*|\b(?:then|do|else|if|elif|while|until)\s+)"
    r"(?:(?:builtin|command|exec|nohup)\s+|time\s+(?:-p\s+)?|!\s*|[A-Za-z_]\w*=[^\s;&|()]*\s+)*"
    r"(?:(?P<back>cd\s+-(?![\w/\\-])|popd\b)"
    # A bare `cd` goes to HOME, which the sandbox envs set to the tool workdir.
    r"|(?P<home>cd(?:\s+(?:-[LPe@]+|--))*\s*(?=[;&|)\n]|$))"
    r"|(?:cd|pushd)\s+(?:(?:-[LPe@]+|--|/d)\s+)*(?P<target>[^\s;&|)]+))",
    re.IGNORECASE,
)


_CHAINED_ON_SUCCESS_RE = re.compile(r"\s*&&")


# Distinct directories, not `cd` commands: padding with repeats must not spend the budget.
_MAX_TRACKED_CWDS = 64
_MAX_WALKED_CWDS = 2048


# The kernel resolves this before any `..`; `$$`/`$BASHPID` name the shell's own PID.
_PROC_CWD_RE = re.compile(r"/proc/(?:self|thread-self|\d+|\$\$|\$\{?BASHPID\}?|\$\{?PPID\}?)/cwd")


def _unquoted_parens(text: str):
    """Yield `(index, char)` for each bracket that is shell syntax, skipping quoted ones.

    `echo '('; cd ../..; echo ')'` writes two brackets no shell ever opens, and counted as syntax
    they ended a subshell that was never entered, dropping the `cd` that came between them.
    """
    quote = ""
    escaped = False
    # A `$(` stays active inside double quotes, so suspend that quote for the subshell.
    suspended: "list[str]" = []
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote:
            if quote == '"' and char == "(" and index and text[index - 1] == "$":
                suspended.append(quote)
                quote = ""
                yield index, char
                continue
            if char == quote:
                quote = ""
            continue
        if char in "'\"":
            quote = char
            continue
        if char in "()":
            yield index, char
            if char == ")" and suspended:
                quote = suspended.pop()


def _subshell_end(text: str, start: int) -> int:
    """Where the subshell containing *start* closes, or the end of *text* when it is not in one.

    `(cd ../..; ls models); cat auth/config.json` runs the `cat` in the unchanged directory: a
    subshell's `cd` dies with the subshell, so its move must not reach past the closing bracket.
    """
    depth = 0
    for index, char in _unquoted_parens(text):
        if index >= start:
            break
        depth = depth + 1 if char == "(" else max(0, depth - 1)
    if not depth:
        return len(text)
    for index, char in _unquoted_parens(text):
        if index < start:
            continue
        depth = depth + 1 if char == "(" else depth - 1
        if char == ")" and not depth:
            return index
    return len(text)


_SHELL_FUNCTION_RE = re.compile(
    r"(?:^|[;&|(){}\n]\s*)(?:function\s+([A-Za-z_][\w.:-]*)\s*(?:\(\s*\))?|([A-Za-z_][\w.:-]*)\s*\(\s*\))\s*\{",
    re.MULTILINE,
)
_MAX_TRACKED_FUNCTIONS = 64


def _unquoted_braces(text: str):
    """Yield `(index, char)` for each `{` or `}` that is shell syntax, skipping quoted ones."""
    quote = ""
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in "'\"":
            quote = char
            continue
        if char in "{}":
            yield index, char


def _uncalled_function_spans(text: str) -> "list[tuple[int, int]]":
    """Spans of function bodies *text* defines but never invokes.

    `helper() { cd ../..; }; cat auth/config.json` never runs the `helper`, so the `cd` in its body
    moves nothing and the `cat` reads the project's own auth file from the unchanged sandbox.
    Refusing that is a false positive, and it is the same rule the python walk already applies to an
    uncalled def.
    """
    bodies: "list[tuple[str, int, int, int]]" = []
    for match in _SHELL_FUNCTION_RE.finditer(text):
        group = 1 if match.group(1) else 2
        name = match.group(group)
        named_at = match.start(group)
        start = text.index("{", match.end() - 1)
        depth = 0
        end = len(text)
        for index, char in _unquoted_braces(text):
            if index < start:
                continue
            depth = depth + 1 if char == "{" else depth - 1
            if not depth:
                end = index
                break
        bodies.append((name, named_at, start, end))
        if len(bodies) >= _MAX_TRACKED_FUNCTIONS:
            break
    spans: "list[tuple[int, int]]" = []
    for name, named_at, start, end in bodies:
        # Assume called unless the name appears nowhere outside its own definition.
        called = any(
            call.start() != named_at and not (start <= call.start() <= end)
            for call in re.finditer(r"(?<![\w.:-])" + re.escape(name) + r"(?![\w.:-])", text)
        )
        if not called:
            spans.append((start, end))
    return spans


def _token_spellings(token: str) -> "list[str]":
    """The paths *token* can be, once the shell has had its say.

    A backslash is a separator on Windows and an escape on POSIX, so `au\\th/auth.db` is both
    `au/th/auth.db` and `auth/auth.db` and the guard has to try each.
    """
    if "\\" not in token:
        return [token]
    return list(dict.fromkeys([token.replace("\\", "/"), token.replace("\\", "")]))


def _path_tokens_containing(text: str, wanted: str):
    """Whole path tokens of *text* holding any character of *wanted* (or the substring `..`).

    One linear pass, then a substring test per token. The equivalent patterns put the interesting
    characters in the MIDDLE, so their leading `[^...]*` backtracks over every position of a long
    token that does not contain one: a fifth of a second for a single 8 KB token, and a command
    carries many.
    """
    for match in _PATH_TOKEN_RE.finditer(text):
        token = match.group(0)
        if ".." in token if wanted == ".." else any(ch in token for ch in wanted):
            yield match


def _cwds_after_cd(workdir: str, text: str) -> "list[tuple[int, int, str]]":
    """Every working directory *text* walks into via `cd`, as `(offset, limit, directory)` in order.

    The offset is where that `cd` ends, because a relative path written BEFORE it opens from the
    old directory: `cat auth/auth.db; cd ../..` reads inside the sandbox and is ordinary work.

    `cd ../..` from the session sandbox lands on the Studio root, and the `auth/auth.db` that
    follows is then the protected database under a name that matches nothing on its own.
    """
    # Possible directories, not one: a failed `cd` leaves the shell where it was. Each state
    # is valid until end of text, or the closing bracket for a subshell move.
    states: "list[tuple[str, int]]" = [(workdir, len(text))]
    walked: "list[tuple[int, int, str]]" = []
    seen: "set[tuple[str, int]]" = set()
    inert = _uncalled_function_spans(text) if "(" in text or "function" in text else []
    # Where each move came from, for `cd -` and `popd`.
    history: "list[list[tuple[str, int]]]" = []
    for match in _DIRECTORY_MOVE_RE.finditer(text):
        returning = match.group("back")
        if returning:
            if any(start <= match.start() <= end for start, end in inert):
                continue
            states = history.pop() if history else [(workdir, len(text))]
            # A return ends every open span here.
            walked = [(offset, min(limit, match.start()), cwd) for offset, limit, cwd in walked]
            # Closed spans: re-entering the same directory is a new span, not a repeat.
            seen.clear()
            # Re-open spans for the directories the return lands in.
            for cwd, until in states:
                if cwd == workdir or until <= match.end() or (cwd, until) in seen:
                    continue
                seen.add((cwd, until))
                walked.append((match.end(), until, cwd))
            continue
        if match.group("home"):
            # A bare `cd` can only land in the sandbox, so it returns like `cd -` (outside subshells).
            if any(start <= match.start() <= end for start, end in inert):
                continue
            if _subshell_end(text, match.end()) != len(text):
                continue
            history.append(states)
            states = [(workdir, len(text))]
            walked = [(offset, min(limit, match.start()), cwd) for offset, limit, cwd in walked]
            seen.clear()
            continue
        target = (match.group("target") or "").strip("'\"")
        if not target or target.startswith("-"):
            continue
        if any(start <= match.start() <= end for start, end in inert):
            continue
        states = [(cwd, until) for cwd, until in states if until >= match.start()] or [
            (workdir, len(text))
        ]
        limit = _subshell_end(text, match.end())
        moved: "list[str]" = []
        for cwd, _until in states:
            nxt = target if os.path.isabs(target) else os.path.normpath(os.path.join(cwd, target))
            if nxt not in moved:
                moved.append(nxt)
            # Deduplicated so repeated `cd .` cannot spend the budget.
            if (nxt, limit) not in seen:
                seen.add((nxt, limit))
                walked.append((match.end(), limit, nxt))
        # Keep both outcomes, capped; the starting directory stays first since padding relies on
        # every `cd` failing. The cap bounds states, never the scan.
        ordered = (
            [(workdir, len(text))]
            + [(c, u) for c, u in states if c != workdir]
            + [(m, limit) for m in moved]
        )
        # After `&&` the first move succeeded, so the unmoved state is not live.
        if _CHAINED_ON_SUCCESS_RE.match(text, match.end()):
            ordered = [(m, limit) for m in moved]
        history.append(states)
        states = list(dict.fromkeys(ordered))[:_MAX_TRACKED_CWDS]
        if len(walked) >= _MAX_WALKED_CWDS:
            break
    return walked


def _references_studio_credential_here(
    text: str,
    workdir: "str | None",
    _unescaped: bool = False,
    _assign_expand_depth: int = 0,
    _quoted_assignments: bool = False,
    _positional_assignments: bool = False,
) -> bool:
    """`_references_studio_credential`, plus the relative paths *text* would open from *workdir*.

    The tool cwd is `<studio home>/sandbox/<session>`, a sibling of the auth directory, so
    `open('../../auth/auth.db')` reads the protected database while naming neither the directory
    nor a credential basename. Resolving only the `..` tokens keeps this to the one shape that can
    leave the sandbox at all."""
    # A command binding the variable itself means its own directory by it.
    if "=" in text and _rebinds_the_studio_home_first(text):
        text = _expand_shell_assignments(text)
    if _references_studio_credential(text):
        return True
    # Naming the root and a credential basename is enough, even unjoined.
    if _CREDENTIAL_BASENAME_RE.search(text) and _text_names_the_studio_root(text):
        return True
    # A recursive read of the root emits credentials without naming them.
    if (
        _text_names_the_studio_root(text)
        and _walks_a_tree_reading_it(text)
        and _names_the_studio_root_itself(text)
    ):
        return True
    # bash removes the backslash in `c\d`, so fold it before the cwd walk.
    if "\\" in text and not _unescaped:
        # One level only: recursing per pass raised RecursionError on long `\\` runs.
        unescaped = _ESCAPED_WORD_CHAR_RE.sub(r"\1", text)
        if unescaped != text and _references_studio_credential_here(
            unescaped, workdir, _unescaped = True
        ):
            return True
    if "[" in text:
        # Read one-char classes as literals before the wildcard collapse discards them.
        literal = _SINGLETON_CLASS_RE.sub(r"\1", text)
        if literal != text and _references_studio_credential_here(literal, workdir):
            return True
    if "[" in text:
        collapsed = _BRACKET_CLASS_RE.sub("?", text)
        if collapsed != text and _references_studio_credential_here(collapsed, workdir):
            return True
    # The kernel resolves the symlink before `..`; lexical normpath would miss it.
    if workdir and "/proc/" in text:
        substituted = _PROC_CWD_RE.sub(lambda _m: workdir.rstrip("/"), text)
        if substituted != text and _references_studio_credential(substituted):
            return True
    # Bypass repoints HOME at the workdir; substitute rather than replace.
    if workdir and ("pwd" in text.lower() or "%cd%" in text.lower()):
        here = _CWD_VARIABLE_RE.sub(lambda _m: workdir.rstrip("/\\"), text)
        if here != text and _references_studio_credential_here(here, workdir):
            return True
    if workdir and ("home" in text.lower() or "~" in text):
        homed = _HOME_VARIABLE_RE.sub(lambda _m: workdir.rstrip("/\\"), text)
        if homed != text and not _HOME_VARIABLE_RE.search(homed):
            if _references_studio_credential_here(homed, workdir):
                return True
    # One level of variable indirection; it only adds detections.
    if "$" in text:
        # Quoted bindings scanned separately: log text shaped like an assignment must not overwrite real ones.
        quoted_modes, quote_states = (_quoted_assignments,), None
        if _assign_expand_depth == 0 and ("'" in text or '"' in text):
            quote_states = _shell_quote_states(text)
            if any(quote_states[m.start(1)] for m in _SHELL_ASSIGN_RE.finditer(text)):
                quoted_modes = (False, True)
        seen = {text}
        for include_quoted in quoted_modes:
            final, positional_text, saw_prefix = _shell_assignment_expansions(
                text, include_quoted = include_quoted, quote_states = quote_states
            )
            if saw_prefix and (_assign_expand_depth == 0 or _positional_assignments):
                positional_text = _shell_assignment_expansions(
                    text, include_quoted = include_quoted, quote_states = quote_states, skip_prefix = True
                )[1]
            variants = (
                ((True, positional_text), (False, final))
                if _assign_expand_depth == 0
                else (
                    (
                        _positional_assignments,
                        positional_text if _positional_assignments else final,
                    ),
                )
            )
            for positional, expanded in variants:
                if expanded in seen:
                    continue
                seen.add(expanded)
                if _assign_expand_depth >= _MAX_SHELL_ASSIGN_EXPAND_PASSES or (
                    "$" in expanded and len(expanded) > max(_MAX_TERMINAL_SCAN_CHARS, len(text))
                ):
                    return True
                if _references_studio_credential_here(
                    expanded,
                    workdir,
                    _assign_expand_depth = _assign_expand_depth + 1,
                    _quoted_assignments = include_quoted,
                    _positional_assignments = positional,
                ):
                    return True
    if workdir and ("cd" in text.lower() or "pushd" in text.lower()):
        for offset, limit, cwd in _cwds_after_cd(workdir, text):
            # The directory itself counts, but only when something runs there.
            if _references_studio_credential(cwd) and text[offset:limit].strip(" \t\n;&|()"):
                return True
            for match in _path_tokens_containing(text, "/\\"):
                # Only paths after the `cd` open from the new directory.
                if match.start() < offset or match.start() >= limit:
                    continue
                token = match.group(0)
                if os.path.isabs(token) or token.startswith("~"):
                    continue
                if any(
                    _references_studio_credential(os.path.normpath(os.path.join(cwd, spelling)))
                    for spelling in _token_spellings(token)
                ):
                    return True
    if not workdir or ".." not in text:
        return False
    for token in (m.group(0) for m in _path_tokens_containing(text, "..")):
        if "/" not in token and "\\" not in token:
            continue
        if any(
            _references_studio_credential(os.path.normpath(os.path.join(workdir, spelling)))
            for spelling in _token_spellings(token)
        ):
            return True
    return False


def _calls_in_uncalled_scopes(tree) -> "set[int]":
    """Ids of calls sitting in a function or lambda body that nothing in *code* calls.

    Deliberately narrow. A body whose name IS called anywhere in the snippet stays live, because the
    move is then real and only the ordering is unknown, and the conservative reading is what a guard
    wants. A branch is not a scope: `if cond: os.chdir(...)` may well run, so it keeps moving.

    A CLASS body is not deferred at all: python executes it when the class statement runs, whether
    or not anything instantiates the class, so `class C: os.chdir("../..")` moves the process. Only
    the methods inside it are deferred, and those are function bodies reached in their own right.
    """
    called: "set[str]" = set()
    # A called or name-bound lambda is not inert.
    invoked: "set[int]" = set()
    lambda_names: "dict[str, int]" = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(node.value, ast.Lambda):
                for target in targets:
                    if isinstance(target, ast.Name):
                        lambda_names[target.id] = id(node.value)
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Lambda):
            invoked.add(id(func))
        elif isinstance(func, ast.Name):
            called.add(func.id)
        elif isinstance(func, ast.Attribute):
            called.add(func.attr)
    invoked.update(node_id for name, node_id in lambda_names.items() if name in called)
    inert: "set[int]" = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in called:
                continue
        elif isinstance(node, ast.Lambda) and id(node) in invoked:
            continue
        elif not isinstance(node, ast.Lambda):
            continue
        # Body only: defaults, decorators and annotations run at definition time.
        body = node.body if isinstance(node.body, list) else [node.body]
        for statement in body:
            for inner in ast.walk(statement):
                if isinstance(inner, ast.Call):
                    inert.add(id(inner))
    return inert


# A recursive copy of the root carries `auth/` along; mirrors the terminal walk rule.
_PY_TREE_COPY_CALLS = frozenset({"copytree", "make_archive", "copy_tree", "unpack_archive"})
# Recursive walks reach `auth/` too; shallow listings read nothing inside it.
_PY_TREE_WALK_CALLS = frozenset("walk rglob glob iglob".split())
_PY_SHALLOW_UNLESS_RECURSIVE = frozenset({"glob", "iglob"})
_PY_TREE_ROOT_CALLS = _PY_TREE_COPY_CALLS | _PY_TREE_WALK_CALLS
# One C-speed scan instead of a substring test per name.
_PY_TREE_CALL_RE = re.compile("|".join(sorted(_PY_TREE_ROOT_CALLS)))


def _python_copies_the_studio_root(tree) -> bool:
    """True when a recursive copy, archive or walk call names the studio root as its SOURCE."""
    root = _studio_home_for_guard()
    if not root:
        return False
    folded_root = _folded_word(root)
    # One binding of the root env var, which neither the env test nor the fold resolves.
    aliases = {
        target.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name not in _PY_TREE_ROOT_CALLS:
            continue
        # Source position per call: copytree first, make_archive third, walkers first or receiver.
        sources = list(node.args[2:4]) if name == "make_archive" else list(node.args[:1])
        if name in _PY_SHALLOW_UNLESS_RECURSIVE and not any(
            isinstance(a, ast.Constant) and isinstance(a.value, str) and "**" in a.value
            for a in node.args
        ):
            continue
        if name in _PY_TREE_WALK_CALLS:
            receiver = getattr(node.func, "value", None)
            sources.append(receiver)
            sources.extend(getattr(receiver, "args", ())[:1])
        sources.extend(
            k.value
            for k in node.keywords
            if k.arg in ("src", "root_dir", "base_dir", "top", "path")
        )
        for argument in sources:
            if isinstance(argument, ast.Name):
                argument = aliases.get(argument.id)
            # The constructor wraps the source name; the fold reduces the call to a marker.
            if isinstance(argument, ast.Call) and getattr(argument, "args", None):
                argument = argument.args[0]
            if isinstance(argument, ast.Name):
                argument = aliases.get(argument.id)
            if argument is None:
                continue
            if _names_the_studio_home_env(argument):
                return True
            folded = _folded_path(argument)
            if isinstance(folded, str) and folded and _folded_word(folded) == folded_root:
                return True
    return False


def _python_builds_a_credential_path(code: str, workdir: "str | None") -> bool:
    """True when a path the CODE builds names the auth directory, however it is spelled.

    `os.path.join("..", "..", "auth", "auth.db")` names the protected database in pieces, so the
    text scan sees four ordinary strings and nothing that looks like a path. `_folded_path` is the
    same fold the sensitive-path analyzer already applies to python; here its result is put through
    the credential test, resolved against the cwd like any other relative path."""
    lowered = code.lower()
    # Cheapest gates first: root nameable, then a tree call, then the real root tests.
    if (
        (
            "studio_home" in lowered
            or any(spelling in lowered for spelling in _studio_home_spellings_lowered())
        )
        and _PY_TREE_CALL_RE.search(lowered)
        and (_text_names_the_studio_root(code) or _code_reads_the_studio_home(code))
    ):
        try:
            if _python_copies_the_studio_root(ast.parse(code)):
                return True
        except (SyntaxError, RecursionError, MemoryError, ValueError):
            pass
    if not any(hint in lowered for hint in _STUDIO_CREDENTIAL_HINTS):
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, RecursionError, MemoryError, ValueError):
        # Syntax errors and unparseable snippets are "nothing to fold", not exceptions.
        return False
    # Source order, since `os.chdir` moves later paths; string constants count too.
    nodes = sorted(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.Call, ast.BinOp, ast.JoinedStr))
            or (isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value)
        ),
        key = lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)),
    )
    # Possible directories: a failed `chdir` may be caught and execution continues.
    chdir_modules = _chdir_modules(tree)
    chdir_names = _chdir_names(tree, chdir_modules)
    # A `chdir` in an uncalled function never runs.
    inert_moves = _calls_in_uncalled_scopes(tree)
    name_bases = _literal_name_bases(tree)
    # `os.open("../..")` fds joined with a later `dir_fd =` open by the kernel.
    dir_fds = _literal_directory_descriptors(tree, workdir)
    studio_env_names, foreign_env_names = _python_env_binding_names(tree)
    process_aliases, process_functions = _process_module_aliases(tree)
    cwds: "list[str | None]" = [workdir]
    # `contextlib.chdir` moves only within its `with` body.
    scoped_until = _scoped_chdir_bounds(tree)
    restore: "list[tuple[int, list]]" = []
    for node in nodes:
        while restore and getattr(node, "lineno", 0) > restore[-1][0]:
            cwds = restore.pop()[1]
        if (
            isinstance(node, ast.Call)
            and _is_chdir_call(node, chdir_names, chdir_modules)
            and id(node) not in inert_moves
        ):
            argument = _chdir_argument(node)
            target = None if argument is None else _folded_path(argument)
            targets: "list[str]" = []
            # `fchdir` has no foldable target; add the studio root as a live state (fail closed).
            if isinstance(node.func, ast.Attribute) and node.func.attr == "fchdir":
                root = _studio_home_for_guard()
                if root:
                    targets = [root]
            # The env var stays in the child, so the move is real; the fold cannot read subscripts.
            if not targets and _names_the_studio_home_env(argument):
                root = _studio_home_for_guard()
                if root:
                    targets = [root]
            if not targets and target and "\x00" not in target:
                # Resolve the parent-walk marker rather than ignoring the move.
                targets = (
                    _parent_walk_targets(argument, target, cwds, name_bases)
                    if "\x02" in target
                    else [target]
                )
            if targets:
                moved: "list[str | None]" = []
                for one in targets:
                    for cwd in cwds:
                        nxt = (
                            one
                            if os.path.isabs(one) or not cwd
                            else os.path.normpath(os.path.join(cwd, one))
                        )
                        if nxt not in moved:
                            moved.append(nxt)
                        if _references_studio_credential(nxt):
                            return True
                bound = scoped_until.get(id(node))
                if bound is not None:
                    restore.append((bound, cwds))
                cwds = (moved + [c for c in cwds if c not in moved])[:_MAX_TRACKED_CWDS]
            continue
        if isinstance(node, ast.Call) and dir_fds and _call_opens_under_a_descriptor(node, dir_fds):
            return True
        if isinstance(node, ast.Call) and _call_runs_from_a_credential_directory(
            node, cwds, name_bases, process_aliases, process_functions
        ):
            # The child opens its arguments from the `cwd` it is given.
            return True
        folded = node.value if isinstance(node, ast.Constant) else _folded_path(node)
        if not folded:
            continue
        if "\x02" in folded:
            # The marker analyzer is skipped in bypass mode, so resolve parents[] here.
            if any(
                _references_studio_credential(target)
                for target in _parent_walk_targets(node, folded, cwds, name_bases)
            ):
                return True
            continue
        if "\x00" in folded:
            # Dynamic pieces belong to the sensitive-path analyzer unless the code reads a studio-home
            # variable, which bypass keeps in the child env.
            root = (
                _studio_home_for_guard()
                if _code_reads_the_studio_home(code)
                and _expression_reads_the_studio_home(node, studio_env_names, foreign_env_names)
                else None
            )
            if root and any(
                _references_studio_credential_here(folded.replace("\x00", root), cwd)
                for cwd in cwds
            ):
                return True
            # Bypass sets PWD to the workdir, so a PWD-based dynamic piece is known too.
            if _code_reads_the_working_directory(code) and any(
                cwd and _references_studio_credential_here(folded.replace("\x00", cwd), cwd)
                for cwd in cwds
            ):
                return True
            continue
        for cwd in cwds:
            if cwd and not os.path.isabs(folded):
                joined = os.path.normpath(os.path.join(cwd, folded.replace("\\", "/")))
                if _references_studio_credential(joined):
                    return True
            if _references_studio_credential_here(folded, cwd):
                return True
    return False


def _parent_chain(node) -> "tuple | None":
    """``(base node, levels walked up)`` for a `.parent` / `.parents[n]` chain, else None."""
    levels = 0
    current = node
    while True:
        if isinstance(current, ast.Attribute) and current.attr == "parent":
            levels += 1
            current = current.value
            continue
        if (
            isinstance(current, ast.Subscript)
            and isinstance(current.value, ast.Attribute)
            and current.value.attr == "parents"
        ):
            index = current.slice
            if isinstance(index, ast.Constant) and isinstance(index.value, int):
                levels += index.value + 1
                current = current.value.value
                continue
            return None
        break
    return (current, levels) if levels else None


def _is_cwd_call(node) -> bool:
    """`Path.cwd()`, `os.getcwd()`, `Path.home()` and the bare spellings of each.

    `home` counts because both subprocess environment builders set `HOME` to the session workdir, so
    `Path.home()` returns the same directory `Path.cwd()` does and a parent walk off it lands in the
    studio root just the same.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    return name in ("cwd", "getcwd", "home", "expanduser")


def _literal_name_bases(tree) -> "dict[str, str]":
    """Names bound once to a literal path, so `root = Path('/tmp/p')` is not read as the cwd.

    A name assigned more than once, or bound to something this fold cannot read, is left out: the
    caller then falls back to the working directory, which is where an unqualified name usually is.
    """
    bases: "dict[str, str]" = {}
    rebound: "set[str]" = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in bases or target.id in rebound:
            rebound.add(target.id)
            bases.pop(target.id, None)
            continue
        value = _folded_path(node.value)
        if value and "\x00" not in value and "\x02" not in value:
            bases[target.id] = value
        else:
            rebound.add(target.id)
    return bases


def _parent_walk_targets(
    node,
    folded: str,
    cwds: "list",
    name_bases: "dict | None" = None,
) -> "list[str]":
    """Where a `.parent` / `.parents[n]` expression can land, resolved rather than guessed.

    The fold writes one marker for the whole walk, so the level count and the base come from the
    expression itself: `Path('/tmp').parent / 'auth'` is `/auth`, not something under the sandbox.
    """
    tail = folded.rsplit("\x02", 1)[-1].lstrip("/\\")
    left = node
    while isinstance(left, ast.BinOp):
        left = left.left
    chain = _parent_chain(left)
    if chain is None:
        return []
    base_node, levels = chain
    bases: "list[str]" = []
    named = (name_bases or {}).get(base_node.id) if isinstance(base_node, ast.Name) else None
    if named:
        bases = (
            [named]
            if os.path.isabs(named)
            else [os.path.normpath(os.path.join(c, named)) for c in cwds if c]
        )
    elif _is_cwd_call(base_node) or isinstance(base_node, ast.Name):
        bases = [c for c in cwds if c]
    else:
        folded_base = _folded_path(base_node)
        if folded_base and "\x00" not in folded_base and "\x02" not in folded_base:
            bases = (
                [folded_base]
                if os.path.isabs(folded_base)
                else [os.path.normpath(os.path.join(c, folded_base)) for c in cwds if c]
            )
    out: "list[str]" = []
    for base in bases:
        for _ in range(min(levels, 64)):
            parent = os.path.dirname(base.rstrip("/\\"))
            if not parent or parent == base:
                break
            base = parent
        out.append(os.path.normpath(os.path.join(base, tail)) if tail else base)
    return out


def _scoped_chdir_bounds(tree) -> dict:
    """`id(call) -> last line of the with body`, for a chdir used as a context manager.

    `with contextlib.chdir(p):` restores the directory on exit, so the move applies to the body and
    nothing after it. A call that is not a `with` item is absent here and stays permanent, which is
    what a bare `os.chdir(p)` does.
    """
    bounds: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        end = getattr(node, "end_lineno", None)
        if end is None:
            continue
        for item in node.items:
            if isinstance(item.context_expr, ast.Call):
                bounds[id(item.context_expr)] = end
    return bounds


# Only these accept a meaningful `cwd`; elsewhere the keyword means anything.
_CHILD_PROCESS_RECEIVERS = {
    "subprocess": frozenset(
        {"run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput"}
    ),
    "asyncio": frozenset({"create_subprocess_exec", "create_subprocess_shell"}),
    "os": frozenset({"popen"}),
}
_CHILD_PROCESS_BARE_NAMES = frozenset(
    {
        "run",
        "Popen",
        "call",
        "check_call",
        "check_output",
        "create_subprocess_exec",
        "create_subprocess_shell",
    }
)


def _process_module_aliases(tree) -> "tuple[dict, set]":
    """`(module aliases, bare process-function names)` for the import forms that rename either one.

    `import subprocess as sp` renames the module and `from subprocess import run as launch` renames
    the function, and both leave the call spelled under a name the fixed tables do not hold.
    """
    aliases: dict = {}
    # Not pre-seeded with run/call/check_output: a snippet's own `def run` is not a launch.
    bare: "set[str]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            functions = _CHILD_PROCESS_RECEIVERS.get(root)
            if not functions:
                continue
            for entry in node.names:
                if entry.name in functions:
                    bare.add(entry.asname or entry.name)
            continue
        if not isinstance(node, ast.Import):
            continue
        for entry in node.names:
            root = entry.name.split(".")[0]
            if root in _CHILD_PROCESS_RECEIVERS and entry.asname:
                aliases[entry.asname] = root
    return aliases, bare


def _launches_a_child_process(
    node: "ast.Call",
    aliases: "dict | None" = None,
    bare: "set | None" = None,
) -> bool:
    """True for a call that starts a process, by module attribute or by a bare imported name.

    The receiver is resolved through the import aliases first. `import subprocess as sp` leaves the
    call spelled `sp.run(...)`, and reading the receiver literally missed it, so a child handed a
    `cwd` outside the sandbox went unchecked.
    """
    func = node.func
    if isinstance(func, ast.Attribute):
        receiver = func.value
        name = receiver.id if isinstance(receiver, ast.Name) else getattr(receiver, "attr", "")
        name = (aliases or {}).get(name, name)
        return func.attr in _CHILD_PROCESS_RECEIVERS.get(name, frozenset())
    return isinstance(func, ast.Name) and func.id in (
        bare if bare is not None else _CHILD_PROCESS_BARE_NAMES
    )


def _literal_directory_descriptors(tree, workdir: "str | None") -> dict:
    """Name -> the directory an `os.open` of a literal path bound to it names.

    Only the shape that can reach the auth directory is resolved: a literal path, opened as a
    descriptor, held in a plain name. Anything else leaves the name absent and the `dir_fd` below
    unresolved, which is what it already was.
    """
    descriptors: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        if not isinstance(value, ast.Call) or not value.args:
            continue
        called = getattr(value.func, "attr", None) or getattr(value.func, "id", None)
        if called != "open":
            continue
        folded = _folded_path(value.args[0])
        if not folded or "\x00" in folded or "\x02" in folded:
            continue
        here = (
            folded
            if os.path.isabs(folded) or not workdir
            else os.path.normpath(os.path.join(workdir, folded))
        )
        for target in targets:
            if isinstance(target, ast.Name):
                descriptors[target.id] = here
    return descriptors


def _call_opens_under_a_descriptor(node: "ast.Call", dir_fds: dict) -> bool:
    """True when a call opens a credential path relative to a tracked directory descriptor."""
    given = next((kw.value for kw in node.keywords if kw.arg == "dir_fd"), None)
    if not isinstance(given, ast.Name):
        return False
    base = dir_fds.get(given.id)
    if not base:
        return False
    for argument in node.args:
        folded = _folded_path(argument)
        if not folded or "\x00" in folded or "\x02" in folded or os.path.isabs(folded):
            continue
        if _references_studio_credential(
            os.path.normpath(os.path.join(base, folded.replace("\\", "/")))
        ):
            return True
    return False


def _call_runs_from_a_credential_directory(
    node: "ast.Call",
    cwds: "list",
    name_bases: "dict",
    process_aliases: "dict | None" = None,
    process_functions: "set | None" = None,
) -> bool:
    """True when a call hands a child process a directory that makes one of its paths a credential.

    `subprocess.run([...], cwd = "../..")` leaves this process where it is, so the walk above never
    moves, and each literal argument was tested against the sandbox instead of against the
    directory the child actually runs from.

    Restricted to the APIs that actually start a process. `cwd` is an ordinary keyword name, and
    reading it as process semantics on any call refused ordinary code:
    `describe("auth/config.json", cwd = "../..")` was blocked in every permission mode even though
    the function may never touch that path.
    """
    if not _launches_a_child_process(node, process_aliases, process_functions):
        return False
    given = next((kw.value for kw in node.keywords if kw.arg == "cwd"), None)
    if given is None:
        return False
    folded = _folded_path(given)
    if (not folded or "\x00" in folded) and _names_the_studio_home_env(given):
        # The fold has no value for the env subscript; resolve it like the chdir walk.
        root = _studio_home_for_guard()
        folded = root or folded
    if not folded or "\x00" in folded:
        return False
    targets = (
        _parent_walk_targets(given, folded, cwds, name_bases) if "\x02" in folded else [folded]
    )
    directories: "list[str]" = []
    for target in targets:
        for cwd in cwds:
            here = (
                target
                if os.path.isabs(target) or not cwd
                else os.path.normpath(os.path.join(cwd, target))
            )
            if here not in directories:
                directories.append(here)
    if any(_references_studio_credential(here) for here in directories):
        return True
    for inner in ast.walk(node):
        if isinstance(inner, ast.Constant) and isinstance(inner.value, str) and inner.value:
            piece = inner.value
        else:
            piece = _folded_path(inner) if isinstance(inner, (ast.BinOp, ast.Call)) else ""
        if not piece or "\x00" in piece or "\x02" in piece or os.path.isabs(piece):
            continue
        for here in directories:
            if _references_studio_credential(
                os.path.normpath(os.path.join(here, piece.replace("\\", "/")))
            ):
                return True
    return False


def _chdir_argument(node: "ast.Call"):
    """The path a `chdir` call is given, positionally or as `path=`."""
    if node.args:
        return node.args[0]
    for keyword in node.keywords:
        if keyword.arg == "path":
            return keyword.value
    return None


def _chdir_names(tree, modules: "set[str] | None" = None) -> "set[str]":
    """Every name that refers to `os.chdir` in this snippet, including aliases.

    `from os import chdir as move` and `move = os.chdir` are both ordinary python, and a walk that
    only knows the literal name resolves everything after `move('../..')` against the wrong place.
    An alias of somebody else's `chdir`, `move = ftp.chdir`, is not one of them.
    """
    modules = modules or {"os", "contextlib"}
    # The bare name counts only once an import binds it; a local `def chdir` is ordinary.
    names: "set[str]" = set()
    defined_locally = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "chdir"
        for node in ast.walk(tree)
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("os", "contextlib"):
            for alias in node.names:
                if alias.name == "chdir":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
            if not isinstance(target, ast.Name):
                continue
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "chdir"
                and isinstance(value.value, ast.Name)
                and value.value.id in modules
            ) or (isinstance(value, ast.Name) and value.id in names):
                names.add(target.id)
    if defined_locally:
        names.discard("chdir")
    return names


def _chdir_modules(tree) -> "set[str]":
    """The names that stand for a module whose `chdir` moves THIS process.

    `ftp.chdir('../..')` changes a remote directory and leaves the local one alone, so reading a
    project's own `auth/config.json` afterwards was refused for a move that never happened.
    """
    # posix/nt are the modules os is built on.
    known = ("os", "contextlib", "posix", "nt")
    modules = set(known)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in known and alias.asname:
                    modules.add(alias.asname)
    return modules


# `os.fchdir(fd)` has an unknown destination, so it is tracked as an unresolved move.
_CHDIR_METHODS = frozenset({"chdir", "fchdir"})


def _is_chdir_call(
    node: "ast.Call",
    names: "set[str] | None" = None,
    modules: "set[str] | None" = None,
) -> bool:
    """True for `os.chdir(...)`, a bare `chdir(...)`, and any alias bound from it.

    `names` is the set of BARE names that refer to it, which is empty for a snippet that defines its
    own `chdir` and never imports one. Empty is meaningful here, so it is distinguished from the
    None the callers that do not compute it pass. The qualified `os.chdir(...)` form is unaffected:
    the receiver is what identifies it there.
    """
    bare = {"chdir"} if names is None else names
    func = node.func
    if isinstance(func, ast.Attribute):
        if func.attr not in _CHDIR_METHODS and func.attr not in bare:
            return False
        return isinstance(func.value, ast.Name) and func.value.id in (
            modules or {"os", "contextlib"}
        )
    return isinstance(func, ast.Name) and func.id in bare


def _reads_an_environment_variable(node) -> bool:
    """True for `os.environ[...]`, `os.environ.get(...)` and `os.getenv(...)`."""
    if isinstance(node, ast.Subscript):
        receiver = node.value
        return (getattr(receiver, "attr", None) or getattr(receiver, "id", None)) == "environ"
    if isinstance(node, ast.Call):
        called = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if called == "getenv":
            return True
        receiver = getattr(node.func, "value", None)
        return (
            called == "get"
            and (getattr(receiver, "attr", None) or getattr(receiver, "id", None)) == "environ"
        )
    return False


def _python_env_binding_names(tree) -> "tuple[set, set]":
    """`(names holding the studio home, names holding some OTHER environment variable)`.

    A dynamic path piece is only the studio root when the expression that built it actually read
    that variable. Attributing it to any snippet that mentions the variable ANYWHERE refused
    `project = os.environ["PROJECT_HOME"]; open(project + "/auth/config.json")`, which names an
    unrelated application's directory.
    """
    studio: "set[str]" = set()
    foreign: "set[str]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        if not _reads_an_environment_variable(value):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                (studio if _names_the_studio_home_env(value) else foreign).add(target.id)
    return studio, foreign


def _expression_reads_the_studio_home(node, studio_names, foreign_names) -> bool:
    """Whether THIS expression's dynamic piece is the studio home.

    Positive attribution both ways: the expression reads the variable itself or uses a name bound to
    it, or every environment-backed name in it belongs to a different variable. Anything this cannot
    attribute keeps the old whole-snippet answer, which is the fail-closed one.
    """
    names = {piece.id for piece in ast.walk(node) if isinstance(piece, ast.Name)}
    if any(_names_the_studio_home_env(piece) for piece in ast.walk(node)):
        return True
    if names & set(studio_names):
        return True
    return not (names & set(foreign_names))


def _variable_points_at_this_install(name: str) -> bool:
    """Whether a studio-home variable's VALUE is this install's root.

    `STUDIO_HOME` is a generic name another application can own, and bypass keeps that foreign value
    in the child, so `open(os.environ["STUDIO_HOME"] + "/auth/config.json")` reads the other
    application. Unset stays True: the child cannot expand it either, so nothing is granted.
    """
    value = (os.environ.get(name.upper()) or "").strip()
    if not value:
        return True
    root = _studio_home_for_guard()
    if not root:
        return True
    try:
        return _same_directory(os.path.expanduser(value), root)
    except Exception:  # noqa: BLE001 - an unreadable value must not break classification
        return True


def _names_the_studio_home_env(node) -> bool:
    """True when *node* reads an environment variable that holds the studio home.

    `os.environ["UNSLOTH_STUDIO_HOME"]`, the `.get` spelling and `os.getenv` all return the same
    directory. Case-insensitive, because `os.environ` upper-cases every key it is handed on Windows.
    """
    if node is None:
        return False
    if isinstance(node, ast.Subscript):
        receiver = node.value
        named = getattr(receiver, "attr", None) or getattr(receiver, "id", None)
        key = node.slice
        return (
            named == "environ"
            and isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and key.value.upper() in _STUDIO_HOME_ENV_VARS
            and _variable_points_at_this_install(key.value)
        )
    if isinstance(node, ast.Call) and node.args:
        called = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        first = node.args[0]
        return (
            called in ("get", "getenv")
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and first.value.upper() in _STUDIO_HOME_ENV_VARS
            and _variable_points_at_this_install(first.value)
        )
    return False


def _code_reads_the_studio_home(code: str) -> bool:
    """True when *code* mentions one of the environment variables that name the studio home.

    Case-insensitive: on Windows `os.environ` upper-cases every key it is given (`os._createenviron`
    sets `encodekey = str.upper` for `nt`), so `os.environ["unsloth_studio_home"]` returns the same
    value the upper-case spelling does. The variable markers the text scan uses are already compared
    lowercased, so this brings the python fold into line with them."""
    lowered = code.lower()
    return any(
        var.lower() in lowered and _variable_points_at_this_install(var)
        for var in _STUDIO_HOME_ENV_VARS
    )


def _code_reads_the_working_directory(code: str) -> bool:
    """True when *code* asks for the working directory by any of its usual names."""
    lowered = code.lower()
    return '"pwd"' in lowered or "'pwd'" in lowered or "getcwd" in lowered or "cwd()" in lowered


_studio_home_lowered_cache: "tuple | None" = None


def _studio_home_spellings_lowered() -> "tuple[str, ...]":
    """The root as source text can spell it, memoized on the marker table it comes from.

    A per-call prefilter, so rebuilding the root for every python snippet showed up. Three forms,
    because a Windows root reaches the text as `c:\\dir`, as `c:\\\\dir` inside a python literal and
    as `c:/dir`; on posix all three are the same string.
    """
    global _studio_home_lowered_cache
    markers = _studio_auth_dir_markers()[0]
    if _studio_home_lowered_cache is None or _studio_home_lowered_cache[0] is not markers:
        # Built from the folded root so stored separators do not decide the answer.
        base = _folded_word(_studio_home_for_guard() or "\x00")
        _studio_home_lowered_cache = (
            markers,
            tuple({base, base.replace("/", "\\"), base.replace("/", "\\\\")}),
        )
    return _studio_home_lowered_cache[1]


def _studio_home_for_guard() -> "str | None":
    """This install's studio root, derived from the same resolution the markers use.

    From the ORIGINAL spelling, not the folded one: a root with an uppercase character in it would
    otherwise be reconstructed in lowercase, and on a case-sensitive host the marker check then
    rejected its own synthesized path, letting a move to the real root through.
    """
    auth_markers, _variable_markers, _cd_re = _studio_auth_dir_markers()
    if not auth_markers:
        return None
    return os.path.dirname(auth_markers[0][2].rstrip("/\\")) or None


def _needs_a_workdir(text: str) -> bool:
    """Whether resolving *text* needs the cwd at all.

    Traversal needs it, and so does anything that MOVES the directory: `pushd <studio home>;
    sqlite3 auth/auth.db` carries no `..`, and without the workdir the walk that would have caught
    it never ran.
    """
    if ".." in text:
        return True
    lowered = text.lower()
    return (
        "cd" in lowered
        or "pushd" in lowered
        or "chdir" in lowered
        # Walks up without `..`; the python analyzer resolves getcwd given the workdir.
        or "getcwd" in lowered
        or "cwd" in lowered
        or "parent" in lowered
    )


def _tool_workdir_for_guard(session_id: "str | None") -> "str | None":
    """The cwd the executor is about to use, or None if it cannot be resolved cheaply."""
    try:
        return _get_workdir(session_id)
    except Exception:  # noqa: BLE001 - an unresolvable workdir leaves the textual match in place
        return None


# `<`/`>` count as leading delimiters (cat <../../notes).
_PARENT_TRAVERSAL_RE = re.compile(r"(?:^|[\s/\\'\"=:<>])\.\.(?:[/\\]|$|[\s'\"])")
# A dynamic segment under a sensitive dir is not provably safe (fail closed).
_SENSITIVE_DIR_RE = re.compile(
    r"/etc/|/(?:var/)?run/secrets[/\\]|(?:^|[/\\])\.(?:ssh|aws|azure|gnupg|docker|kube)[/\\]"
    r"|(?:^|[/\\])\.config/(?:gcloud|gh)[/\\]",
    re.IGNORECASE,
)
# /etc/./passwd and /etc//passwd resolve to /etc/passwd.
_REDUNDANT_SLASH_RE = re.compile(r"/\.?(?=/)")
# Substitute assigned values to catch substring tricks (p=passwd; cat /etc/${p:0:6}).
_SHELL_VAR_RE = re.compile(r"\$\{(\w+)(?::[^{}]*)?\}|\$(\w+)")
# Apply ${p/X/w} replacements before scanning.
_SHELL_PARAM_REPL_RE = re.compile(r"\$\{(\w+)/(/)?([^/{}]*)/([^{}]*)\}")
# Apply case modification (${p,,}) before scanning.
_SHELL_PARAM_CASE_RE = re.compile(r"\$\{(\w+)(\^\^|,,|\^|,)\}")
# Resolve indirect expansion ${!p}.
_SHELL_PARAM_INDIRECT_RE = re.compile(r"\$\{!(\w+)\}")
_SHELL_PARAM_VALUE_OP_RE = re.compile(r"\$\{([A-Za-z_]\w*)(:?)([-=+])([^{}]*)\}")
_SHELL_ASSIGN_POSITION_CHARS = frozenset(";&|(\n'\"`{")
_SHELL_ASSIGN_KEYWORDS = frozenset(
    ("export", "local", "declare", "typeset", "readonly", "then", "do", "else", "{", "!", "time")
)
_SHELL_ASSIGN_RE = re.compile(r"(?:^|[\s;&|(])([A-Za-z_]\w*)=([^\s;&|)]*)")
# bash expands $'...' after this classifier, so decode first.
_ANSI_C_RE = re.compile(r"\$'((?:[^'\\]|\\.)*)'")
# bash concatenates quoted pieces (enviro''n), so strip quotes before the scan.
_SHELL_QUOTE_RE = re.compile(r"['\"]")
_GLOB_BRACKET_RE = re.compile(r"\[([^!\]][^\]]*)\]")
# fnmatch does not understand POSIX classes; each matches one char, so use `?`.
_POSIX_CLASS_RE = re.compile(r"\[\[:\w+:\]\]")
# Sensitive files a glob could expand to (cat /e??/passwd).
_SENSITIVE_GLOB_TARGETS = tuple(
    _joined(parts)
    for parts in (
        ("/etc/pas", "swd"),
        ("/etc/sh", "adow"),
        ("/etc/sudoers",),
        ("/root/.ssh/id_r", "sa"),
        ("/root/.aws/cred", "entials"),
        ("/home/u/.ssh/id_r", "sa"),
        ("/home/u/.ssh/id_ed", "25519"),
        ("/home/u/.aws/cred", "entials"),
        ("/home/u/.net", "rc"),
        ("/home/u/.git-cred", "entials"),
    )
)
# Every file under these is a credential, so any glob into them asks.
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
# Credential basenames a glob can reach anywhere (cat ~/.netr?).
_SENSITIVE_GLOB_BASENAMES = frozenset(
    _joined(parts)
    for parts in (
        ("token",),
        ("stored_tokens",),
        ("cred", "entials"),
        (".net", "rc"),
        ("net", "rc"),
        (".pypirc",),
        (".npmrc",),
        (".git-cred", "entials"),
        ("id_r", "sa"),
        ("id_ed", "25519"),
        ("id_ec", "dsa"),
        ("id_d", "sa"),
        ("pas", "swd"),
        ("sh", "adow"),
        # A project .env holds secrets; globs to it are gated like the literal path.
        (".env",),
    )
)
_REDIR_PREFIX_RE = re.compile(r"^\d*[<>]+")
# bash brace expansion runs after this classifier; expand to scan each result.
_BRACE_COMMA_RE = re.compile(r"^\{([^{}]*,[^{}]*)\}$")
_BRACE_SEQ_RE = re.compile(r"^\{([^{}]+)\.\.([^{}]+)(?:\.\.(-?\d+))?\}$")
_BRACE_ANY_RE = re.compile(r"\{[^{}]*,[^{}]*\}|\{[^{}]+\.\.[^{}]+(?:\.\.-?\d+)?\}")
# Substitute ${x:-passwd} operands so the resulting path is scanned.
_SHELL_PARAM_OP_RE = re.compile(r"\$\{[A-Za-z_]\w*:?[-=+]([^{}]*)\}")


# The credential regex is superlinear; overlong text fails closed.
_MAX_PATH_SCAN_CHARS = 2048
_MAX_TERMINAL_SCAN_CHARS = 4096
_MAX_SHELL_ASSIGN_EXPAND_PASSES = 16
# Where the memoised node list is parked on a parsed tree (see _tree_nodes).
_TREE_NODES_ATTR = "_unsloth_walk_nodes"


@functools.lru_cache(maxsize = 2048)
def _token_command_base(token: str) -> str:
    """The command name a shell token would run: separators stripped, directory dropped, folded to
    lower case.

    Several scans recompute this for every token of every command, and the tokens repeat heavily
    (`cat`, `-la`, `|`), so the result is memoised. Pure function of the token text.
    """
    return os.path.basename(token.strip(";&|()`{}")).lower()


@functools.lru_cache(maxsize = 64)
def _parse_python(code: str):
    """``(tree, None)`` or ``(None, SyntaxError)`` for a snippet, parsed at most once.

    Classifying one python tool call parses the same source two or three times over: the safety
    check parses it, the classifier parses it again, and the ``python -c`` path parses it a third
    time before delegating. The tree is only ever read, so one parse serves all of them. The cache
    is bounded and keyed on the source text, so it cannot go stale.
    """
    try:
        return ast.parse(code), None
    except SyntaxError as exc:
        return None, exc


def _tree_nodes(tree) -> list:
    """``list(ast.walk(tree))``, computed once per parsed tree.

    The python classifiers each sweep the whole tree a dozen times looking for a different node
    shape, and every ``ast.walk`` rebuilds the same traversal from scratch. The node list is
    identical for all of them, so it is built once and parked on the tree itself: the classifiers
    only read the AST, and each call parses its own tree, so nothing can go stale. Falls back to a
    plain walk if the attribute cannot be set.
    """
    nodes = getattr(tree, _TREE_NODES_ATTR, None)
    if nodes is None:
        nodes = list(ast.walk(tree))
        try:
            setattr(tree, _TREE_NODES_ATTR, nodes)
        except (AttributeError, TypeError):
            pass
    return nodes


def _references_sensitive_path(text: str) -> bool:
    """True if a command or string literal reads a credential path or escapes the sandbox workdir
    via parent traversal."""
    if len(text) > _MAX_PATH_SCAN_CHARS:
        return True
    if _PARENT_TRAVERSAL_RE.search(text) or _SENSITIVE_PATH_RE.search(text):
        return True
    # Re-scan only when a rewrite actually changed the text.
    norm = _REDUNDANT_SLASH_RE.sub("", text)
    if norm != text and _SENSITIVE_PATH_RE.search(norm):
        return True
    debracket = _GLOB_BRACKET_RE.sub(lambda m: m.group(1)[0], text)
    return bool(
        _references_studio_credential(text)
        or (debracket != text and _SENSITIVE_PATH_RE.search(debracket))
    )


def _pattern_matches_dir(pattern: str, target: str) -> bool:
    """Segment-wise fnmatch so a glob segment does not cross a '/' boundary (`/home/*` must not
    match `/home/u/.ssh`)."""
    p = pattern.split("/")
    t = target.split("/")
    if len(p) != len(t):
        return False
    return all(fnmatch.fnmatch(tseg, pseg) for pseg, tseg in zip(p, t))


def _glob_token_sensitive(token: str) -> bool:
    """True if a single ? / * / [..] glob token could expand to a sensitive file or a file under a
    secret/credential directory. Shared by the terminal scan and the Python glob check."""
    # Rewrites never add a glob metacharacter, so a token without one cannot match.
    if not _GLOB_META_RE.search(token):
        return False
    token = _REDIR_PREFIX_RE.sub("", _SHELL_QUOTE_RE.sub("", token))
    token = _POSIX_CLASS_RE.sub("?", token)
    if not any(c in token for c in "?*["):
        return False
    if any(fnmatch.fnmatch(target, token) for target in _SENSITIVE_GLOB_TARGETS):
        return True
    base = token.rsplit("/", 1)[-1]
    if any(c in base for c in "?*[") and any(
        fnmatch.fnmatch(name, base) for name in _SENSITIVE_GLOB_BASENAMES
    ):
        return True
    head = token.rsplit("/", 1)[0] if "/" in token else token
    return any(
        _pattern_matches_dir(token, d) or _pattern_matches_dir(head, d)
        for d in _SENSITIVE_GLOB_DIRS
    )


def _glob_hits_sensitive(command: str) -> bool:
    """True if any glob token in a command could expand to a sensitive file, so `cat /e??/passwd`
    asks even without a literal sensitive path."""
    if not _GLOB_META_RE.search(command):
        return False
    return any(
        _glob_token_sensitive(token)
        for token in command.replace(";", " ").replace("|", " ").split()
    )


# `C:x` is drive-relative, so the separator is required for an absolute path.
_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_WIN_DRIVE_RELATIVE_JOIN_RE = re.compile(r"^[A-Za-z]:(?![\\/:])[^:\s]+$")


def _posix_join(parts) -> str:
    """Join folded path pieces the way ``os.path.join`` and ``Path(...)`` actually resolve them.

    An absolute component DISCARDS everything before it: `os.path.join("/usr", "/media/x")` opens
    `/media/x`, not `/usr/media/x`. Doing this at the join keeps the distinction from a doubled
    separator inside one literal, which the OS simply collapses -- `/home/alice//usr/report.txt` is
    `/home/alice/usr/report.txt` and has nothing to do with `/usr`. Inferring the join from `//`
    after the fact could not tell those apart, and read `/usr` out of the literal.
    """
    out = ""
    for part in parts:
        if not out:
            out = part
        elif _WIN_DRIVE_RE.match(part) or _WIN_DRIVE_RELATIVE_JOIN_RE.match(part):
            # A drive-qualified operand discards the left, as ntpath.join does.
            out = part
        elif part[:2] == "\\\\":
            out = part
        elif part[:1] in ("/", "\\"):
            # A leading backslash is rooted on the current drive: keep the drive, drop the rest.
            drive = _WIN_DRIVE_RE.match(out)
            out = (out[:2] if drive else "") + part
        else:
            out = out.rstrip("/") + "/" + part
    return out


def _shell_assign_value_self_references(name: str, value: str) -> bool:
    """True when *value* expands *name* (VAR=$VAR), which must not feed back into itself."""
    if "$" not in value:
        return False
    if any((m.group(1) or m.group(2)) == name for m in _SHELL_VAR_RE.finditer(value)):
        return True
    return any(
        m.group(1) == name
        for pattern in (
            _SHELL_PARAM_REPL_RE,
            _SHELL_PARAM_CASE_RE,
            _SHELL_PARAM_INDIRECT_RE,
            _SHELL_PARAM_VALUE_OP_RE,
        )
        for m in pattern.finditer(value)
    )


def _expand_shell_assignments(
    command: str,
    *,
    _include_quoted: bool = True,
    _positional: bool = False,
) -> str:
    """Best-effort substitution of `NAME=value ... $NAME`, so a sensitive path split across an
    assignment and an argument (p=/etc; cat $p/passwd) is still visible to the scan. Also applies
    pattern replacement. Fail-open: only adds detections."""
    final, positional, _ = _shell_assignment_expansions(command, include_quoted = _include_quoted)
    return positional if _positional else final


def _shell_assignment_expansions(
    command: str,
    *,
    include_quoted: bool = True,
    quote_states = None,
    skip_prefix: bool = False,
) -> "tuple[str, str, bool]":
    """(last binding everywhere, binding active at each use, saw a command-prefix assignment).

    `x=/tmp cat "$x"` expands the argument with the OUTER x and only hands /tmp to the child, so with
    *skip_prefix* such assignments bind nothing; the last-binding result keeps them for the child."""
    env = {}
    saw_prefix = False
    inert_states = None

    def repl_default(m):
        name, colon, op, operand = m.groups()
        value = env.get(name)
        missing = value is None or (colon and not value.strip("'\""))
        if op == "+":
            return "" if missing else operand
        return operand if missing else value

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
        pointed = env.get(m.group(1))
        return env.get(pointed, m.group(0)) if pointed is not None else m.group(0)

    def expand(text):
        if "$" not in text:
            return text
        text = _SHELL_PARAM_INDIRECT_RE.sub(repl_indirect, text)
        text = _SHELL_PARAM_REPL_RE.sub(repl_pattern, text)
        text = _SHELL_PARAM_CASE_RE.sub(repl_case, text)
        return _SHELL_VAR_RE.sub(lambda m: env.get(m.group(1) or m.group(2), m.group(0)), text)

    pieces, pos = [], 0
    matches = list(_SHELL_ASSIGN_RE.finditer(command))
    prefix = [False] * len(matches)
    for i in range(len(matches) - 1, -1, -1):
        m = matches[i]
        if (quote_states is None or not quote_states[m.start(1)]) and (
            _assignment_is_a_command_prefix(command, m.start(2))
        ):
            chained = (
                i + 1 < len(matches) and not command[m.end(2) : matches[i + 1].start(1)].strip()
            )
            prefix[i] = prefix[i + 1] if chained else True
    for i, m in enumerate(matches):
        if prefix[i] or (quote_states is not None and quote_states[m.start(1)]):
            continue
        j = m.start(1) - 1
        while j >= 0 and command[j] in " \t":
            j -= 1
        if j < 0 or command[j] in _SHELL_ASSIGN_POSITION_CHARS:
            continue
        if i and matches[i - 1].end(2) == j + 1:
            prefix[i] = prefix[i - 1]
            continue
        k = j
        while k >= 0 and command[k] not in " \t;&|(\n":
            k -= 1
        if command[k + 1 : j + 1] not in _SHELL_ASSIGN_KEYWORDS:
            prefix[i] = True
    for i, match in enumerate(matches):
        if not include_quoted and ("'" in command or '"' in command):
            if quote_states is None:
                quote_states = _shell_quote_states(command)
            if quote_states[match.start(1)]:
                continue
        var, val = match.groups()
        pieces.append(expand(command[pos : match.start(2)]))
        pieces.append(expand(val))
        pos = match.end(2)
        if prefix[i]:
            saw_prefix = True
            if skip_prefix:
                continue
        if _shell_assign_value_self_references(var, val):
            # Studio home vars stay references: `H=$H; cat "$H/auth/auth.db"` must still name the install.
            if var.upper() in _STUDIO_HOME_ENV_VARS:
                env.setdefault(var, "${" + var + "}")
            val = _SHELL_PARAM_VALUE_OP_RE.sub(repl_default, val)
            env.setdefault(var, "")
            val = expand(val)
            if len(val) > _MAX_PATH_SCAN_CHARS:
                continue
        if not val and var in env:
            if inert_states is None:
                inert_states = _assignment_inert_states(command)
            if inert_states[match.start(1)]:
                continue
        env[var] = val
    if not env:
        return command, command, saw_prefix
    return expand(command), "".join(pieces) + expand(command[pos:]), saw_prefix


def _expand_param_defaults(command: str) -> str:
    """Substitute the operand of a default/alternate parameter expansion (cat /etc/pass${x:-wd}),
    which bash applies after this classifier. Fail-open: only adds detections."""
    return _SHELL_PARAM_OP_RE.sub(lambda m: m.group(1), command)


# Separators inside $'...' are data; callers neutralize them before tokenizing.
_ANSI_C_SEPARATOR_RE = re.compile(r"[\s;&|()<>`]")
# Must not be read as a command start by the boundary regex in _find_blocked_commands.
_ANSI_C_NEWLINE_MARK = "\x03"
_ANSI_C_NEWLINE_RE = re.compile(r"[\n\r]")


def _folded_str_literal(node) -> "str | None":
    """The string an expression evaluates to when built only from string literals ("un" + "link"),
    else None. Resolves a name spelled dynamically but fully known at parse time."""
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
    """Decode bash ANSI-C quoted words (cat $'/etc/pass\x77d') so an escape-obfuscated path is
    visible to the scan. Fail-open: only adds detections. With ``keep_one_word`` the decoded text
    cannot introduce new shell syntax, which is what bash does with it."""

    def dec(m):
        try:
            text = bytes(m.group(1), "utf-8").decode("unicode_escape")
        except (UnicodeDecodeError, ValueError):
            return m.group(0)
        if not keep_one_word:
            return text
        if _ANSI_C_NEWLINE_MARK not in text:
            # Re-quote rather than flatten: bash passes ONE word, and newlines end sed comments. The
            # newline becomes a mark so the boundary regex does not read a new command there.
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
    """Best-effort bash brace expansion so a sensitive path split across a brace group is scanned.
    Bounded. Fail-open: only detects."""
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
        return False
    if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
        return bool(_PY_WRITE_MODE_RE.search(mode_node.value))
    return True


def _has_kwarg_splat(node) -> bool:
    """True if the call has a ``**kwargs`` splat, which can hide a write mode."""
    return any(kw.arg is None for kw in node.keywords or [])


def _builtin_open_writes(node) -> bool:
    """Write check for builtin ``open(file, mode)`` (mode is the 2nd arg)."""
    if _has_kwarg_splat(node):
        return True
    if any(isinstance(a, ast.Starred) for a in node.args):
        return True
    mode = node.args[1] if len(node.args) >= 2 else None
    for kw in node.keywords or []:
        if kw.arg == "mode":
            mode = kw.value
    return _mode_arg_writes(mode)


def _attr_open_writes(node) -> bool:
    """Write check for ``x.open(...)`` (e.g. ``Path.open(mode)`` where mode is the 1st arg). Only a
    mode-looking string is read as the mode, so a ``ZipFile.open("name.txt")`` read is not
    mistaken for a write."""
    if _has_kwarg_splat(node):
        return True
    for kw in node.keywords or []:
        if kw.arg == "mode":
            return _mode_arg_writes(kw.value)
    if node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            if _PY_MODE_LITERAL_RE.match(first.value):
                return bool(_PY_WRITE_MODE_RE.search(first.value))
            # A 2nd positional is a mode or os.open flags; honor a string mode, otherwise ask.
            if len(node.args) >= 2:
                second = node.args[1]
                if isinstance(second, ast.Constant) and isinstance(second.value, str):
                    return _mode_arg_writes(second)
                return True
            return False
        return True
    return False


_PATH_CTORS = (
    "Path",
    "PurePath",
    "PurePosixPath",
    "PureWindowsPath",
    "PosixPath",
    "WindowsPath",
)
# Pass-through normalizers keep the location, so fold through them.
_PATH_PASSTHROUGH_ATTRS = frozenset(
    {"abspath", "normpath", "realpath", "expanduser", "expandvars", "resolve", "absolute"}
)
# Rewrite only the last component (Path('/etc/x').with_name('passwd')).
_PATH_NAME_REWRITES = frozenset({"with_name", "with_stem", "with_suffix"})
_PERCENT_NAMED_RE = re.compile(r"%\((\w+)\)[-#0 +]*\d*(?:\.\d+)?[a-zA-Z]")


def _folded_path(
    node,
    literals = None,
    ctors = None,
    join_names = None,
) -> "str | None":
    """Best-effort value of a path built from string literals, so a sensitive path assembled from
    pieces (os.path.join('/etc', 'passwd'), Path('/etc') / 'passwd', f'/etc/{name}') is still
    visible to the scan. A dynamic piece becomes NUL, a non-slash placeholder, so a dynamic segment
    under a sensitive dir is still detectable.

    ``literals`` maps names bound to string literals; ``ctors`` is the set of pathlib constructor
    names; ``join_names`` are bare names bound to os.path.join.
    """
    literals = literals or {}
    ctors = ctors or _PATH_CTORS
    join_names = join_names or frozenset()

    def fold(node) -> "str | None":
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            return (
                node.value.decode("latin-1", "ignore")
                if isinstance(node.value, bytes)
                else node.value
            )
        if isinstance(node, ast.Name):
            return literals.get(node.id)
        if isinstance(node, ast.Attribute) and node.attr in ("parent", "parents"):
            # `.parent`/`.parents` escape the workdir without '..'; \x02 is the escape sentinel.
            return "\x02"
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and (node.value.attr == "parents")
        ):
            return "\x02"
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
            # `/` is a JOIN: an absolute right side discards the left, as os.path.join does.
            return _posix_join((left, right)) if isinstance(node.op, ast.Div) else left + right
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            template = fold(node.left)
            if template is not None and "%" in template:
                rhs = node.right
                if "%(" in template:
                    # Unresolved values or non-literal mappings leave the NUL marker (fail closed).
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
                # joinpath uses the POSIX join rule: an absolute piece discards everything left of it.
                base = fold(func.value)
                parts = [base if base is not None else "\x00"]
                parts += [(fold(a) or "\x00") for a in node.args]
                return _posix_join(parts)
            if isinstance(func, ast.Attribute) and func.attr in ("glob", "rglob", "iglob"):
                base = fold(func.value)
                pattern = fold(node.args[0]) if node.args else "\x00"
                return (base if base is not None else "\x00") + "/" + (pattern or "\x00")
            if isinstance(func, ast.Attribute) and func.attr in _PATH_NAME_REWRITES:
                # Unresolved receiver stays None; a dynamic arg becomes the NUL marker.
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
                else:
                    name = stem + arg
                return head + name
            if isinstance(func, ast.Attribute) and func.attr in _PATH_PASSTHROUGH_ATTRS:
                return fold(node.args[0]) if node.args else fold(func.value)
            if isinstance(func, ast.Attribute) and func.attr == "join":
                # str.join has the separator as receiver; tell it apart from os.path.join(*pieces).
                sep = fold(func.value)
                if (
                    sep is not None
                    and len(node.args) == 1
                    and isinstance(node.args[0], (ast.List, ast.Tuple))
                ):
                    pieces = [(fold(e) or "\x00") for e in node.args[0].elts]
                    # str.join concatenates; only os.path.join/pathlib let an absolute piece reset.
                    return sep.join(pieces)
                parts = [(fold(a) or "\x00") for a in node.args]
                return _posix_join(parts)
            if isinstance(func, ast.Name) and func.id in join_names:
                parts = [(fold(a) or "\x00") for a in node.args]
                return _posix_join(parts)
            if (isinstance(func, ast.Attribute) and func.attr in ctors) or (
                isinstance(func, ast.Name) and func.id in ctors
            ):
                parts = [(fold(a) or "\x00") for a in node.args]
                return _posix_join(parts)
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
    """True if a folded path with a dynamic piece (NUL) inside a path segment could spell a
    credential target, e.g. open('/et' + chr(99) + '/passwd'). NUL matches any run of
    non-separator chars so the dynamic split of a sensitive name resolves, while an all-dynamic
    or segment-spanning path cannot form a single credential name and stays safe."""
    if not folded or "\x00" not in folded:
        return False
    pattern = "".join(r"[^/\\]*" if ch == "\x00" else re.escape(ch) for ch in folded)
    try:
        rx = re.compile(pattern + r"\Z")
    except re.error:
        return True
    return any(rx.match(t) for t in _SENSITIVE_GLOB_TARGETS)


def _folded_is_sensitive(folded) -> bool:
    """A folded path is sensitive if it names a credential file, has a dynamic segment (NUL)
    directly under a sensitive directory, walks out of the sandbox via a pathlib .parent escape
    (\x02), or is a glob that could resolve to a credential path."""
    if not folded:
        return False
    return (
        "\x02" in folded
        or _references_sensitive_path(folded)
        or ("\x00" in folded and bool(_SENSITIVE_DIR_RE.search(folded)))
        # A dynamic piece can be the "/" of a sensitive root (os.sep + "etc/passwd").
        or ("\x00" in folded and _references_sensitive_path(folded.replace("\x00", "/")))
        # A dynamic piece inside a name ('/et' + chr(99)): match literals around each NUL.
        or _dynamic_name_hits_sensitive(folded)
        or _glob_token_sensitive(folded)
    )


def _command_references_sensitive(command: str) -> bool:
    """True if a shell command reads/writes a credential path or escapes the sandbox workdir (../),
    after undoing the shell expansions that would hide it: quotes/backslash escapes,
    brace/parameter/ANSI-C expansion and NAME=value prefixes."""
    stripped = _SHELL_QUOTE_RE.sub("", command).replace("\\", "")
    # A set drops identical candidates so the superlinear pattern runs once per distinct text.
    candidates = set()
    for c in (command, stripped, _decode_ansi_c(command)):
        c_param = _expand_param_defaults(c)
        candidates.update((c, c_param, _expand_braces(c_param), _expand_shell_assignments(c_param)))
    return any(_glob_hits_sensitive(c) or _references_sensitive_path(c) for c in candidates)


_CMD_ECHO_OFF_RE = re.compile(r"(?<![^\s&|()])@+")
_CMD_CONTROL_RE = re.compile(
    r"(?i)(?<![^\s&|()])(?:if\s+(?:/i\s+)?(?:not\s+)?(?:(?:exist|defined|errorlevel|cmdextversion)\s+\S+"
    r"|\S+?\s*==\s*\S+|\S+\s+(?:equ|neq|lss|leq|gtr|geq)\s+\S+)|do|call)(?=\s)"
)


def _cmd_reading(command: str) -> str:
    """How cmd splits a command for the POSIX classifiers: ' is an ordinary character, ^ only escapes,
    a leading @ only turns the echo off, and IF / FOR ... DO / CALL run the command that follows."""
    text = _CMD_ECHO_OFF_RE.sub("", command.replace("^", "").replace("'", " "))
    return _CMD_CONTROL_RE.sub(" & ", text)


# The request's sandbox level while a call is classified: Low runs the Terminal on the host shell,
# so the classifier must not probe for (or assume) the isolated cmd profile.
_classifying_sandbox_level: "ContextVar[str | None]" = ContextVar(
    "unsloth_classifying_sandbox_level", default = None
)


@contextlib.contextmanager
def classifying_under(sandbox_level: "str | None"):
    token = _classifying_sandbox_level.set(sandbox_level)
    try:
        yield
    finally:
        _classifying_sandbox_level.reset(token)


# Both run the command through cmd.exe: the isolated one inside MXC, the fallback on a host without Git Bash.
_CMD_PROFILES = ("cmd_isolated", "cmd_fallback")


def _reads_differently_under_cmd(command: str) -> bool:
    """True when cmd.exe will run ``command`` and would split it unlike bash."""
    return (
        sys.platform == "win32"
        and _cmd_reading(command) != command
        and _terminal_profile(_classifying_sandbox_level.get() == "low") in _CMD_PROFILES
    )


def _terminal_is_potentially_unsafe(command: str) -> bool:
    """Classify a terminal command for auto mode (fail closed)."""
    if not command or not command.strip():
        return False
    if _reads_differently_under_cmd(command) and _terminal_is_potentially_unsafe(
        _cmd_reading(command)
    ):
        return True
    # A quoted ">" false-positives into a prompt, which is the safe direction.
    if ">" in command or "`" in command or "$(" in command or "<(" in command:
        return True
    if _command_references_sensitive(command):
        return True
    # shlex reads newlines as whitespace, which would demote "ls\nrm x" to arguments.
    command = command.replace("\r\n", ";").replace("\n", ";").replace("\r", ";")
    try:
        lexer = shlex.shlex(command, posix = True, punctuation_chars = ";&|()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return True
    # Re-lex the expanded command so assigned/default roots are seen.
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
    # find/fd `(...)` resets command context, so scan every token.
    if any(_token_command_base(t) in ("find", "fd") for t in scan_tokens):
        if any(t.split("=", 1)[0] in _AUTO_UNSAFE_FIND_LIKE_FLAGS for t in scan_tokens):
            return True
    # Absolute operands outside the silent roots reach the user's filesystem. Run unconditionally:
    # gating on `/` or `~` would skip Windows spellings and attached redirections.
    if _terminal_reaches_outside_sandbox(tokens, command) or (
        scan_tokens is not tokens and _terminal_reaches_outside_sandbox(scan_tokens, command)
    ):
        return True
    if any(t.startswith("/") or t.startswith("~") for t in scan_tokens):
        token_bases = [_token_command_base(t) for t in tokens]
        if any(b in _AUTO_RECURSIVE_SEARCH or b in _AUTO_RECURSIVE_LISTERS for b in token_bases):
            return True
        # ls only walks the subtree with -R.
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
    for _tok_idx, token in enumerate(tokens):
        # Runs like ";;" lex as one token and still separate commands.
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
            # Matches `--output=x`, attached `-o/tmp/out` and clusters (`sort -uo out`).
            flag_head = token.split("=", 1)[0]
            cluster = token[1:] if token[:2] != "--" and "=" not in token else ""
            # GNU accepts unambiguous long-option abbreviations; prefixes fail closed.
            is_long_abbrev = flag_head.startswith("--") and len(flag_head) > 2
            for uf in _AUTO_UNSAFE_COMMAND_FLAGS.get(current_command, ()):
                if flag_head == uf or (len(uf) == 2 and (token.startswith(uf) or uf[1] in cluster)):
                    return True
                if is_long_abbrev and uf.startswith("--") and uf.startswith(flag_head):
                    return True
            # Consume the flag's value so it is not read as a positional.
            pending_flag_value = "=" not in token and (
                (current_command == "date" and flag_head in _DATE_DISPLAY_VALUE_FLAGS)
                or flag_head in _SECOND_POSITIONAL_VALUE_FLAGS.get(current_command, ())
            )
            if not prefix_pending:
                expect_command = False
            continue
        if not expect_command:
            raw_pos = token.strip(";&|()`{}")
            # Count file positionals; the second one is written.
            if current_command in _AUTO_SECOND_POSITIONAL_WRITES:
                if pending_flag_value:
                    pending_flag_value = False
                elif raw_pos:
                    positional_args += 1
                    if positional_args >= 2:
                        return True
            # A positional past display-flag values sets state (date's +FORMAT stays read-only).
            elif current_command in _AUTO_ARG_SENSITIVE_COMMANDS:
                if pending_flag_value:
                    pending_flag_value = False
                elif raw_pos and not (current_command == "date" and raw_pos.startswith("+")):
                    return True
            continue
        if _ASSIGNMENT_RE.match(token):
            if _env_assignment_is_unsafe(token.split("=", 1)[0]):
                return True
            continue
        if prefix_pending and token.lstrip("-").isdigit():
            continue
        raw = token.strip(";&|()`{}")
        # A path-qualified command is an arbitrary executable, not the trusted utility.
        if "/" in raw or "\\" in raw:
            return True
        base = os.path.basename(raw).lower()
        stem, ext = os.path.splitext(base)
        if ext in {".exe", ".com", ".bat", ".cmd"}:
            base = stem
        if base in _AUTO_SAFE_WRAPPERS:
            prefix_pending = True
            # Track the wrapper so its own flags are checked; the real command overwrites this.
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


def _is_literal_false(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def _is_inert_loader_arg(node: ast.AST, mmap_slot: bool) -> bool:
    # False, None, or a string in load's mmap_mode slot.
    return isinstance(node, ast.Constant) and (
        node.value is None or node.value is False or (mmap_slot and isinstance(node.value, str))
    )


def _python_is_potentially_unsafe(code: str) -> bool:
    """Classify python-tool code for auto mode (fail closed)."""
    if not code or not code.strip():
        return False
    # Surface what execution would refuse as a confirmation first.
    if _check_code_safety(code) is not None:
        return True
    tree, _parse_error = _parse_python(code)
    if _parse_error is not None:
        return False
    # Kept in step with the high-risk gate so both classifiers agree on fs scope.
    if _python_reaches_outside_sandbox(tree, code):
        return True
    open_aliases = {"open"}
    attr_open_aliases: "set[str]" = set()
    builtins_aliases = {"builtins", "__builtins__"}
    dynamic_aliases = set()
    # compile() builds a code object that FunctionType/exec can run.
    code_exec_aliases = {"exec", "eval", "__import__", "breakpoint", "compile"}
    literal_str_vars: "dict[str, str]" = {}
    path_ctor_aliases = set(_PATH_CTORS)
    pathjoin_aliases: "set[str]" = set()
    writer_aliases: "set[str]" = set()
    os_aliases = {"os", "posix"}
    load_module_aliases = set(_AUTO_UNSAFE_PY_LOAD_MODULES)
    # numpy module aliases, and names / attributes bound to a numpy pickle loader -> its allow_pickle position.
    numpy_aliases = {"numpy"}
    pickle_fn_aliases: "dict[str, int]" = {}
    pickle_fn_attr_aliases: "dict[str, int]" = {}
    # Names bound to the builtin getattr (g = getattr), so a dynamic lookup aliased through it still fails closed.
    getattr_aliases = {"getattr"}
    partial_aliases: "set[str]" = set()
    archive_ctor_aliases: "set[str]" = set()
    # operator.methodcaller is dynamic dispatch, like getattr.
    operator_aliases = {"operator"}
    methodcaller_aliases: "set[str]" = set()
    basicconfig_aliases: "set[str]" = set()
    fileinput_aliases = {"fileinput"}
    # The write-callable gate keeps map(len, ...) safe.
    invoker_aliases = set(_HIGHER_ORDER_INVOKERS)

    def _in_numpy(node) -> bool:
        while isinstance(node, ast.Attribute):
            node = node.value
        return isinstance(node, ast.Name) and node.id in numpy_aliases

    def _allow_pickle_position(node) -> "int | None":
        # A conditional or boolean callee counts if any branch is a loader.
        if isinstance(node, ast.NamedExpr):
            return _allow_pickle_position(node.value)
        if isinstance(node, (ast.IfExp, ast.BoolOp)):
            branches = [node.body, node.orelse] if isinstance(node, ast.IfExp) else node.values
            return next(
                (pos for pos in map(_allow_pickle_position, branches) if pos is not None), None
            )
        if isinstance(node, ast.Name):
            return pickle_fn_aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            if node.attr in pickle_fn_attr_aliases:
                return pickle_fn_attr_aliases[node.attr]
            if node.attr in _NUMPY_PICKLE_FLAG_POS and _in_numpy(node.value):
                return _NUMPY_PICKLE_FLAG_POS[node.attr]
        return None

    def _is_dynamic_namespace(node) -> bool:
        # Namespace lookups (globals(), __dict__, __builtins__, sys.modules) are as dynamic as getattr.
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
        if not call.args:
            return False
        first = call.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            return True
        return first.value in _AUTO_UNSAFE_PY_ATTRS or first.value in _AUTO_UNSAFE_PY_WRITE_METHODS

    def _fileinput_inplace(call) -> bool:
        if _has_kwarg_splat(call):
            return True
        for kw in call.keywords or []:
            if kw.arg == "inplace":
                v = kw.value
                if isinstance(v, ast.Constant):
                    return bool(v.value)
                return True
        return False

    def _basicconfig_writes(call) -> bool:
        if _has_kwarg_splat(call):
            return True
        return any(kw.arg == "filename" for kw in call.keywords or [])

    def _wraps_write_callable(arg) -> bool:
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
        # Omits the dynamic/getattr/code-exec poison aliases, which are gated where called.
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

    # Poison multiply-bound names: every assignment is visited before calls, so a later benign
    # reassignment would mask an earlier sensitive value.
    assign_counts: "dict[str, int]" = {}
    for node in _tree_nodes(tree):
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
    for node in _tree_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "builtins":
                    builtins_aliases.add(alias.asname or "builtins")
                elif alias.name in ("os", "posix"):
                    os_aliases.add(alias.asname or alias.name)
                elif alias.name in _AUTO_UNSAFE_PY_LOAD_MODULES:
                    load_module_aliases.add(alias.asname or alias.name)
                elif alias.name.split(".")[0] == "numpy":
                    numpy_aliases.add(alias.asname or "numpy")
                elif alias.name == "operator":
                    operator_aliases.add(alias.asname or "operator")
                elif alias.name == "fileinput":
                    fileinput_aliases.add(alias.asname or "fileinput")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "numpy":
                for alias in node.names:
                    if alias.name == "*":
                        pickle_fn_aliases.update(_NUMPY_PICKLE_FLAG_POS)
                    elif alias.name in _NUMPY_PICKLE_FLAG_POS:
                        pickle_fn_aliases[alias.asname or alias.name] = _NUMPY_PICKLE_FLAG_POS[
                            alias.name
                        ]
                    else:
                        numpy_aliases.add(alias.asname or alias.name)
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
                if alias.name in _HIGHER_ORDER_INVOKERS:
                    invoker_aliases.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            value = node.value
            if isinstance(node, ast.AnnAssign):
                assign_targets = [node.target]
            else:
                assign_targets = node.targets
            targets = [t.id for t in assign_targets if isinstance(t, ast.Name)]
            attr_targets = [t.attr for t in assign_targets if isinstance(t, ast.Attribute)]
            if _in_numpy(value):
                numpy_aliases.update(targets)
            if (_pos := _allow_pickle_position(value)) is not None:
                pickle_fn_aliases.update(dict.fromkeys(targets, _pos))
                pickle_fn_attr_aliases.update(dict.fromkeys(attr_targets, _pos))
            if isinstance(value, ast.Name) and value.id in open_aliases:
                open_aliases.update(targets)
                attr_open_aliases.update(attr_targets)
            elif isinstance(value, ast.Name) and value.id in getattr_aliases:
                getattr_aliases.update(targets)
            elif isinstance(value, ast.Name) and value.id in partial_aliases:
                partial_aliases.update(targets)
            elif isinstance(value, ast.Name) and value.id in writer_aliases:
                writer_aliases.update(targets)
            elif isinstance(value, ast.Name) and value.id in archive_ctor_aliases:
                archive_ctor_aliases.update(targets)
            elif isinstance(value, ast.Name) and value.id in invoker_aliases:
                invoker_aliases.update(targets)
            elif isinstance(value, ast.Name) and value.id in path_ctor_aliases:
                path_ctor_aliases.update(targets)
            elif isinstance(value, ast.Name) and value.id in pathjoin_aliases:
                pathjoin_aliases.update(targets)
            elif isinstance(value, ast.Attribute) and value.attr == "join":
                pathjoin_aliases.update(targets)
            elif isinstance(value, ast.Attribute) and value.attr in _PATH_CTORS:
                path_ctor_aliases.update(targets)
            elif (
                isinstance(value, ast.Attribute)
                and value.attr == "open"
                and isinstance(value.value, ast.Name)
                and value.value.id in builtins_aliases
            ):
                open_aliases.update(targets)
            elif (
                isinstance(value, ast.Attribute)
                and value.attr in code_exec_aliases
                and isinstance(value.value, ast.Name)
                and value.value.id in builtins_aliases
            ):
                code_exec_aliases.update(targets)
            elif isinstance(value, ast.Attribute) and value.attr in _AUTO_UNSAFE_PY_WRITE_METHODS:
                writer_aliases.update(targets)
            elif isinstance(value, ast.Attribute) and value.attr == "open":
                # A captured .open's mode position varies, so fail closed on the call.
                dynamic_aliases.update(targets)
            elif isinstance(value, ast.Attribute) and value.attr in _ARCHIVE_CTOR_NAMES:
                archive_ctor_aliases.update(targets)
            elif isinstance(value, ast.Subscript):
                dynamic_aliases.update(targets)
            elif (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in getattr_aliases
            ):
                dynamic_aliases.update(targets)
            elif (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr in ("get", "pop", "setdefault")
                and _is_dynamic_namespace(value.func.value)
            ):
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
                dynamic_aliases.update(targets)
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
                dynamic_aliases.update(targets)
            elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                for t in targets:
                    literal_str_vars[t] = "\x02" if t in multi_assigned_names else value.value
            elif isinstance(value, (ast.Call, ast.BinOp, ast.Name, ast.JoinedStr)):
                folded = _folded_path(value, literal_str_vars, path_ctor_aliases, pathjoin_aliases)
                if folded is not None and "\x00" not in folded and "\x02" not in folded:
                    for t in targets:
                        literal_str_vars[t] = "\x02" if t in multi_assigned_names else folded
            elif isinstance(value, (ast.Tuple, ast.List)):
                # Destructuring binds elements like single assignments, both callables and literals.
                for target in assign_targets:
                    if isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == len(
                        value.elts
                    ):
                        for tgt_el, val_el in zip(target.elts, value.elts):
                            if (
                                isinstance(tgt_el, ast.Attribute)
                                and (_pos := _allow_pickle_position(val_el)) is not None
                            ):
                                pickle_fn_attr_aliases[tgt_el.attr] = _pos
                            if not isinstance(tgt_el, ast.Name):
                                continue
                            tid = tgt_el.id
                            if (_pos := _allow_pickle_position(val_el)) is not None:
                                pickle_fn_aliases[tid] = _pos
                            if isinstance(val_el, ast.Name) and val_el.id in open_aliases:
                                open_aliases.add(tid)
                            elif isinstance(val_el, ast.Name) and val_el.id in getattr_aliases:
                                getattr_aliases.add(tid)
                            elif isinstance(val_el, ast.Name) and val_el.id in partial_aliases:
                                partial_aliases.add(tid)
                            elif isinstance(val_el, ast.Name) and val_el.id in writer_aliases:
                                writer_aliases.add(tid)
                            elif isinstance(val_el, ast.Name) and val_el.id in archive_ctor_aliases:
                                archive_ctor_aliases.add(tid)
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
            # Defaults align to the tail of posonlyargs+args; kw_defaults align 1:1 with kwonlyargs.
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
                if (_pos := _allow_pickle_position(_default)) is not None:
                    pickle_fn_aliases[_param.arg] = _pos
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
                    dynamic_aliases.add(_param.arg)
    # Presence, not dataflow: loaders can be aliased, subclassed or returned in endless ways.
    _module_names = set(_AUTO_UNSAFE_PY_LOAD_MODULES)
    _bare_names = set(_AUTO_UNSAFE_YAML_LOADERS)
    _imported_modules = set()
    for node in _tree_nodes(tree):
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
                    _module_names.add(alias.asname or alias.name)
    _module_names |= _imported_modules

    def _loader_receiver(node) -> bool:
        while isinstance(node, ast.Attribute):
            node = node.value
        return isinstance(node, ast.Name) and node.id in _module_names

    for node in _tree_nodes(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in _AUTO_UNSAFE_YAML_LOADERS:
                return True
            if (
                node.attr in _AUTO_UNSAFE_PY_LOAD_ATTRS or node.attr in _AUTO_UNSAFE_PY_LOAD_CLASSES
            ) and _loader_receiver(node.value):
                return True
        elif isinstance(node, ast.Name) and node.id in _bare_names:
            return True
        # Returning the module hands out every loader on it.
        elif isinstance(node, (ast.Return, ast.Lambda)):
            _out = node.value if isinstance(node, ast.Return) else node.body
            if isinstance(_out, ast.Name) and _out.id in _imported_modules:
                return True
    try:
        for node in _tree_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _AUTO_UNSAFE_PY_MODULES:
                        return True
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in _AUTO_UNSAFE_PY_MODULES:
                    return True
                for alias in node.names:
                    if alias.name == "*" or alias.name in _AUTO_UNSAFE_PY_ATTRS:
                        return True
                    if alias.name == "open" and node.module in ("os", "posix"):
                        return True
            elif isinstance(node, ast.Attribute):
                # Any reference fails closed, even uncalled (rm = os.remove; rm("x")).
                if node.attr in _AUTO_UNSAFE_PY_ATTRS:
                    return True
                # z = np.load("a.npz"); z.allow_pickle = True turns pickling on for the next z["x"].
                if node.attr == "allow_pickle" and isinstance(node.ctx, ast.Store):
                    return True
                # builtins.exec / eval / __import__ (and compile/breakpoint) are dynamic code execution, matching the
                # bare-name code_exec_aliases path; __builtins__.__import__(...) is a dynamic import that dodges the
                # static import check.
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
                val = node.value
                if isinstance(val, bytes):
                    val = val.decode("latin-1", "ignore")
                if isinstance(val, str) and (
                    _references_sensitive_path(val) or _glob_token_sensitive(val)
                ):
                    return True
            elif isinstance(node, (ast.BinOp, ast.JoinedStr)):
                if _folded_is_sensitive(
                    _folded_path(node, literal_str_vars, path_ctor_aliases, pathjoin_aliases)
                ):
                    return True
            elif isinstance(node, ast.Call):
                if _folded_is_sensitive(
                    _folded_path(node, literal_str_vars, path_ctor_aliases, pathjoin_aliases)
                ):
                    return True
                func = node.func
                # Unwrap x.__call__(args) so it reaches the checks below.
                if isinstance(func, ast.Attribute) and func.attr == "__call__":
                    func = func.value
                if isinstance(func, (ast.Call, ast.Subscript)):
                    return True  # calling a call/subscript result is dynamic
                # numpy allow_pickle unpickles like torch.load: only a literal False (None/False positionally) is safe.
                if any(
                    kw.arg == "allow_pickle" and not _is_literal_false(kw.value)
                    for kw in node.keywords
                ) or (
                    (_flag_pos := _allow_pickle_position(func)) is not None
                    and (
                        not all(
                            _is_inert_loader_arg(arg, mmap_slot = i == 1 and _flag_pos == 2)
                            for i, arg in enumerate(node.args[1:3], start = 1)
                        )
                        or any(isinstance(arg, ast.Starred) for arg in node.args)
                        or any(kw.arg is None for kw in node.keywords)
                    )
                ):
                    return True
                # A concrete write callable handed as an argument to any call escapes into a helper that can invoke it
                # without a direct open()/writer site: the same bypass the map/starmap/reduce branches gate, but
                # through a user-defined helper. A benign callable argument (run(len)) is unaffected.
                if any(_passed_write_callable(a) for a in node.args) or any(
                    _passed_write_callable(kw.value) for kw in node.keywords
                ):
                    return True
                # A numpy loader handed to a helper can be called there with allow_pickle positionally.
                if any(_allow_pickle_position(a) is not None for a in node.args) or any(
                    _allow_pickle_position(kw.value) is not None for kw in node.keywords
                ):
                    return True
                if isinstance(func, ast.Name):
                    if func.id in dynamic_aliases:
                        return True
                    if func.id in open_aliases and _builtin_open_writes(node):
                        return True
                    if func.id in writer_aliases:
                        return True
                    if func.id in archive_ctor_aliases and _builtin_open_writes(node):
                        return True
                    if func.id in basicconfig_aliases and _basicconfig_writes(node):
                        return True
                    # A writer passed to map/filter runs without a direct call site.
                    if (
                        func.id in invoker_aliases
                        and node.args
                        and _wraps_write_callable(node.args[0])
                    ):
                        return True
                elif isinstance(func, ast.Attribute):
                    if func.attr in _AUTO_UNSAFE_PY_WRITE_METHODS:
                        return True
                    if func.attr == "basicConfig" and _basicconfig_writes(node):
                        return True
                    if (
                        func.attr in _HIGHER_ORDER_INVOKERS
                        and node.args
                        and _wraps_write_callable(node.args[0])
                    ):
                        return True
                    if (
                        func.attr == "input"
                        and isinstance(func.value, ast.Name)
                        and func.value.id in fileinput_aliases
                        and _fileinput_inplace(node)
                    ):
                        return True
                    if (
                        func.attr == "open"
                        and isinstance(func.value, ast.Name)
                        and func.value.id in os_aliases
                    ):
                        return True
                    if (
                        func.attr == "load"
                        and isinstance(func.value, ast.Name)
                        and func.value.id in load_module_aliases
                    ):
                        return True
                    if func.attr == "open" and _attr_open_writes(node):
                        return True
                    if func.attr in attr_open_aliases and _builtin_open_writes(node):
                        return True
                    if func.attr in _ARCHIVE_CTOR_NAMES and _builtin_open_writes(node):
                        return True
                    # Enumerating a directory outside the sandbox reveals host filenames; dynamic dirs are left
                    # to other checks.
                    _enum_dir = None
                    if func.attr == "iterdir":
                        _enum_dir = func.value
                    elif func.attr in ("glob", "rglob", "iglob"):
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
        return True
    return False


# Mirrors the sandbox SSRF blocklist.
_MCP_METADATA_HOST_RE = re.compile(
    r"169\.254\.\d{1,3}\.\d{1,3}|"
    r"100\.100\.100\.\d{1,3}|"
    r"fd00:ec2::254|"
    r"metadata\.google\.internal|"
    r"metadata\.tencentyun\.com|"
    r"://metadata(?=[:/])",
    re.IGNORECASE,
)


_MCP_CREDENTIAL_KEY_RE = re.compile(
    r"^(?:authorization|proxy-authorization|cookie|set-cookie|"
    r"x-api-key|api[-_]?key|apikey|x-auth-token|auth[-_]?token|access[-_]?token|"
    r"refresh[-_]?token|id[-_]?token|bearer|private[-_]?key|secret[-_]?key|"
    r"client[-_]?secret|password|passwd|session[-_]?token)$",
    re.IGNORECASE,
)


def _mcp_arguments_reference_studio_credential(arguments) -> bool:
    """True if an MCP call's arguments point at Studio's auth directory. An MCP server runs outside
    the terminal sandbox, so a filesystem server would read the credential the local tools refuse.
    Prose fields are skipped for the same reason they are below: an issue body that mentions the
    filename is text to store, not a file to open."""

    def walk(value, is_prose: bool = False) -> bool:
        if isinstance(value, str):
            return False if is_prose else _references_studio_credential(value)
        if isinstance(value, dict):
            return any(
                walk(v, is_prose or (isinstance(k, str) and k.lower() in _MCP_PROSE_KEYS))
                for k, v in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(walk(v, is_prose) for v in value)
        return False

    return walk(arguments)


def _mcp_arguments_reference_sensitive(arguments) -> bool:
    """True if any string in an MCP call's arguments names a credential path, a credential/secret
    environment variable, or a cloud-metadata host."""

    def key_is_credential(key) -> bool:
        return isinstance(key, str) and bool(_MCP_CREDENTIAL_KEY_RE.match(key.strip()))

    def walk(value, is_prose: bool = False) -> bool:
        if isinstance(value, str):
            # Skip prose keys rather than allowlisting path keys: a path can ride any name.
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


_SQL_DDL_OBJECTS = (
    r"table|database|schema|index|view|function|procedure|trigger|"
    r"sequence|role|user|extension|type|domain|aggregate|policy"
)
_SQL_DDL_MODIFIERS = (
    r"(?:(?:or\s+replace|unique|temp|temporary|global|local|materialized|recursive)\s+)*"
)
_SQL_IDENT = r'(?:\w+|"(?:[^"]|"")*"|`(?:[^`]|``)*`|\[[^\]]+\])'
_SQL_UPDATE_TARGET = r"(?:only\s+)?" + _SQL_IDENT + r"(?:\s*\.\s*" + _SQL_IDENT + r")*"
# Whole statements only, so prose containing "delete" stays safe.
_MCP_ARG_MUTATION_RE = re.compile(
    r"\b(?:delete\s+from|"
    r"drop\s+" + _SQL_DDL_MODIFIERS + r"(?:" + _SQL_DDL_OBJECTS + r")|"
    # Match the whole identifier, or the trailing \b lets TRUNCATE users slip through.
    r"truncate\s+(?:table\s+)?[\"\[`]?\w+|"
    # Implicit aliases are left out: indistinguishable from prose "update x y set".
    r"update\s+" + _SQL_UPDATE_TARGET + r"(?:\s+as\s+" + _SQL_IDENT + r")?\s+set\b|"
    r"insert\s+into|replace\s+into|"
    # Bare SELECT INTO is left out (PL/pgSQL reads into a variable with it).
    r"select\s+[^;]*?\binto\s+(?:outfile|dumpfile)\b|"
    r"alter\s+system\b|"
    r"alter\s+" + _SQL_DDL_MODIFIERS + r"(?:" + _SQL_DDL_OBJECTS + r")|"
    r"create\s+" + _SQL_DDL_MODIFIERS + r"(?:" + _SQL_DDL_OBJECTS + r")|"
    r"grant\s+\w+|revoke\s+\w+|merge\s+into|"
    r"comment\s+on\b|security\s+label\b|lock\s+table\b|"
    r"refresh\s+materialized\s+view|reindex\s+\w+|"
    # CALL needs "(", ";" or end so "call me back" stays safe.
    r"call\s+\w+(?=\s*[(;]|\s*$)|exec(?:ute)?\s+\w+|vacuum|"
    r"copy\s+[^;]*?\b(?:from|to)\b)\b",
    re.IGNORECASE,
)
# SQLite: ATTACH/DETACH, write-form PRAGMA, load_extension().
_MCP_ARG_SQLITE_MUTATION_RE = re.compile(
    r"\b(?:attach|detach)\s+database\b"
    r"|\battach\s+(?:database\s+)?['\"]"
    r"|\bpragma\s+\w+(?:\.\w+)?\s*(?:=|\()"
    r"|\bload_extension\s*\(",
    re.IGNORECASE,
)
# The trailing "(" is required so a column like setval_count stays safe.
_MCP_ARG_SQL_FUNCTION_RE = re.compile(
    r"\b(?:pg_terminate_backend|pg_cancel_backend|pg_write_file|lo_export|"
    r"lo_import|setval|nextval|set_config|pg_notify|dblink_exec|pg_reload_conf|"
    r"pg_rotate_logfile|"
    r"pg_advisory_(?:lock|lock_shared|unlock|unlock_shared|unlock_all|"
    r"xact_lock|xact_lock_shared)|"
    r"pg_try_advisory_(?:lock|lock_shared|xact_lock|xact_lock_shared))\s*\(",
    re.IGNORECASE,
)
# SQL comments are whitespace (DELETE/**/FROM), so collapse them first.
_SQL_COMMENT_RE = re.compile(r"/\*.*?\*/|--[^\n]*", re.DOTALL)
# Directives may sit between the name and body.
_GRAPHQL_MUTATION_RE = re.compile(
    r"\bmutation\b\s*\w*\s*(?:@\w+(?:\s*\([^)]*\))?\s*)*[({]", re.IGNORECASE
)
_GRAPHQL_COMMENT_RE = re.compile(r"#[^\n]*")


_MUTATING_HTTP_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_HTTP_METHOD_KEYS = frozenset({"method", "http_method", "httpmethod", "verb", "http_verb"})


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
    """True if an MCP call's arguments carry a mutating command, so a read-named but write-capable
    tool asks."""

    def walk(value, in_query: bool = False) -> bool:
        if isinstance(value, str):
            # A mention of DELETE FROM in prose is not a statement this call runs.
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


# render_html runs arbitrary HTML/JS and can egress under the preview CSP when artifact network
# access is on. Relative, url(#id) and data: refs are not matched.
_RENDER_HTML_NETWORK_RE = re.compile(
    r"\bfetch\s*\(|"
    r"XMLHttpRequest|"
    r"\bWebSocket\b|"
    r"\bEventSource\b|"
    r"\bsendBeacon\b|"
    r"\bimportScripts\b|"
    r"navigator\s*\.\s*serviceWorker|"
    # Workers run scripts this static scan cannot see.
    r"\bnew\s+(?:Shared)?Worker\s*\(|"
    r"@import|"
    r"url\(\s*[\"']?\s*(?:https?:|/)|"
    r"<script[^>]*\bsrc\s*=|"
    r"\b(?:src|href|srcset)\s*=\s*[\"']?\s*(?:https?:|/)|"
    # Navigation sinks; reload()/history.back do not navigate to a new URL.
    r"\blocation\s*\.\s*(?:assign|replace)\s*\(|"
    r"\bwindow\s*\.\s*open\s*\(|"
    r"\b(?:window\s*\.\s*)?location(?:\s*\.\s*href)?\s*=\s*[\"'`]?\s*(?:https?:|/)|"
    r"\[\s*[\"'](?:fetch|open|XMLHttpRequest|WebSocket|EventSource|importScripts|"
    r"sendBeacon|serviceWorker)[\"']\s*\]|"
    # Anchored to location so str['replace'](...) stays static.
    r"(?:\blocation|\[\s*[\"']location[\"']\s*\])\s*\[\s*[\"'](?:assign|replace)[\"']\s*\]\s*\(|"
    r"(?:\blocation|\[\s*[\"']location[\"']\s*\])\s*\[\s*[\"']href[\"']\s*\]"
    r"\s*=\s*[\"'`]?\s*(?:https?:|/)|"
    # Anchored to a host object so a plain obj['a'+'b'] stays safe.
    r"\b(?:window|self|globalThis|top|parent|frames)\s*\[[^\]]*"
    r"(?:[\"']\s*\+|\+\s*[\"'])[^\]]*\]|"
    # Meta-refresh to a URL; a bare self-reload has no url=.
    r"<meta\b(?=[^>]*http-equiv\s*=\s*[\"']?\s*refresh)(?=[^>]*\burl\s*=)|"
    r"\bwss?://",
    re.IGNORECASE,
)
# Line comments are kept: stripping them would eat the // in https:// URLs.
_JS_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _render_html_reaches_network(arguments: dict) -> bool:
    code = arguments.get("code")
    if not isinstance(code, str):
        return False
    return bool(_RENDER_HTML_NETWORK_RE.search(_JS_BLOCK_COMMENT_RE.sub("", code)))


# deep_research runs nothing; without it here the unknown-name default would prompt.
_ALWAYS_SAFE_TOOLS = frozenset(
    {
        "web_search",
        "search_knowledge_base",
        "search_conversation",
        "read_skill",
        "deep_research",
        "mcp_tool_schema",
        "view_image",
    }
)


def never_needs_approval(name: str) -> bool:
    """search_conversation only reads this chat's own compacted turns (#11671)."""
    return name == "search_conversation"


def is_always_safe_tool(name: str) -> bool:
    """True for tools that never need an auto-mode prompt on any arguments, so a caller (e.g. the
    streaming provisional card) can allow them before the full arguments are known. render_html
    is intentionally excluded: a networked canvas needs approval, which cannot be judged before
    its arguments stream."""
    return name in _ALWAYS_SAFE_TOOLS


_TEXT_PREVIEW_TOOLS = frozenset({"python", "terminal", "edit_file"})


def has_text_only_provisional_card(name: str) -> bool:
    """True when streaming this tool's arguments before approval shows only text. A large code
    payload takes a minute or more to write, and suppressing the card until the call completes
    leaves the chat blank the whole time. Nothing runs before the decision either way."""
    return name in _TEXT_PREVIEW_TOOLS


def _web_search_fetches_url(name: str, arguments: dict) -> bool:
    """web_search carrying a ``url`` fetches that exact page instead of searching, which is egress
    to a host the CALL names, so it asks even though plain search stays always-safe. Name-only
    ``is_always_safe_tool`` is deliberately unchanged: it runs before arguments exist, where a
    query-only search must not prompt."""
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
    if name == "render_html":
        return _render_html_reaches_network(arguments)
    if name.startswith(MCP_TOOL_PREFIX):
        tool_name = _mcp_raw_tool_name(name)
        if tool_name in _BLENDER_CLI_SUMMARY_TOOLS:
            return True
        tool_name = _MCP_TERM_SEPARATOR_RE.sub("_", tool_name)
        if _AUTO_UNSAFE_MCP_VERB_RE.search(tool_name):
            return True
        if _AUTO_SENSITIVE_MCP_NOUN_RE.search(tool_name):
            return True
        if _mcp_arguments_reference_sensitive(arguments):
            return True
        if _mcp_arguments_mutate(arguments):
            return True
        return not _AUTO_SAFE_MCP_TOOL_RE.match(tool_name)
    if name == "terminal":
        return _terminal_is_potentially_unsafe(str(arguments.get("command", "")))
    if name == "python":
        return _python_is_potentially_unsafe(str(arguments.get("code", "")))
    # Always writes; stated explicitly so it cannot become the quiet way around python's prompt.
    if name == "edit_file":
        return True
    return True


# High risk regardless of arguments; ordinary dev commands run. The hard blocks, rlimits and
# env scrubbing still apply beneath this prompt.
_HIGH_RISK_COMMANDS = frozenset(
    {
        "sudo",
        "su",
        "doas",
        "pkexec",
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
        # cmd.exe built-ins reachable via the `cmd /c` fallback.
        "del",
        "erase",
        "rd",
        "kill",
        "pkill",
        "killall",
        "taskkill",
        "tskill",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "setcap",
        "crontab",
        # atd runs the payload later, outside this call's blocklist, rlimits and cancellation.
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
        "iptables",
        "ip6tables",
        "nft",
        "ufw",
        "mount",
        "umount",
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
        "unlink",
        "format",
        "diskpart",
        "diskutil",
        # Gated wholesale: the destructive subcommand lives in the arguments.
        "systemd-run",
        "schtasks",
        "reg",
        "sc",
        "launchctl",
        # Container daemons act with host privileges; chroot/nsenter/unshare hide a nested command.
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
_SYSCTL_WRITE_FLAGS = frozenset({"-w", "--write", "-p", "--load", "--system"})
# Transparent only for the high-risk scan, where its privilege flags are gated on their own.
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
# Destroys contents in place; plain -l allocation only grows a file.
_FALLOCATE_DESTRUCTIVE_FLAGS = frozenset(
    {"-p", "--punch-hole", "-z", "--zero-range", "-c", "--collapse-range", "-d", "--dig-holes"}
)
_HIGH_RISK_RECURSIVE_COMMANDS = frozenset({"chmod", "chown", "chgrp"})
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
# Tracers and profilers run the rest of the line as a child.
_TRACER_LAUNCHERS = frozenset({"strace", "ltrace", "ktrace", "dtruss", "perf", "valgrind"})
_EXEC_FLAG_FORWARDING_COMMANDS = frozenset({"find", "fd"})
_EXEC_FORWARD_FLAGS = frozenset(
    {"-exec", "-execdir", "-ok", "-okdir", "--exec", "--exec-batch", "-x", "-X"}
)
_ATTACHED_EXEC_FLAGS = frozenset({"-exec", "-execdir", "--exec", "--exec-batch"})
_HIGH_RISK_FIND_FLAGS = frozenset({"-delete"})
# Flag values that are commands the tool executes (tar --checkpoint-action, rsync -e).
_HIGH_RISK_ARG_EXEC_FLAGS = frozenset({"--checkpoint-action", "--rsh", "--rsync-path"})
# Only for the owning utilities, so a mere mention does not prompt.
_ARG_EXEC_FLAG_OWNERS = frozenset({"tar", "gtar", "bsdtar", "rsync", "scp", "sftp"})
# No network namespace, so a listener exposes the workdir. Position-scoped: `pip install
# uvicorn` starts nothing.
_LISTENER_PY_MODULES = (
    r"http\.server|SimpleHTTPServer|uvicorn|gunicorn|waitress|flask|"
    r"twisted|websockets|aiohttp\.web"
)
_LISTENER_PY_MODULE_RE = re.compile(
    r"(?:^|[;&|\n(]|&&|\|\|)\s*(?:[A-Za-z_]\w*=\S*\s+)*(?:\S*/)?"
    r"(?:python|pypy)[0-9.]*\s+(?:-\S+\s+)*-m\s+(?:" + _LISTENER_PY_MODULES + r")\b",
    re.IGNORECASE,
)
# Matched after wrapper resolution (`env python -m http.server`).
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
# Short forms may be attached (-d@f), so they match prefix-wise.
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
# POST is omitted: already caught by the body/upload flags.
_WGET_METHOD_FLAGS = frozenset({"--method"})
_CURL_METHOD_FLAGS = frozenset({"-X", "--request"})
_CURL_DESTRUCTIVE_METHODS = frozenset({"delete", "put", "patch"})
# Separate from curl's so wget -T/-F are not misread as uploads.
_WGET_UPLOAD_FLAGS = frozenset({"--post-data", "--post-file", "--body-data", "--body-file"})
_PIPE_TO_INTERPRETER_RE = re.compile(
    r"\|\s*(?:sudo\s+)?(?:sh|bash|zsh|dash|ksh|fish|python[0-9.]*|node|ruby|perl|php)\b"
)
_BARE_TRUNCATING_REDIRECT_RE = re.compile(r"(?:^|[;&|\n(]|&&|\|\|)\s*(?::|true)?\s*>(?!>)\s*\S")
_HERESTRING_TO_INTERPRETER_RE = re.compile(
    r"\b(?:sh|bash|zsh|dash|ksh|fish|ash|python[0-9.]*|node|ruby|perl|php)\b[^\n]*<<<"
)
# Generated script content is unscreenable; diff <(...) only reads and stays out.
_PROC_SUBST_EXEC_RE = re.compile(
    r"\b(?:sh|bash|zsh|dash|ksh|fish|ash|source|eval|python[0-9.]*|node|nodejs|bun|ruby|perl|php)\b"
    r"[^\n]*<\("
    r"|(?:^|[;&|\n(]|&&|\|\|)\s*\.\s+<\("
)
# No network namespace; command position only, so filename arguments are not misread.
_NETWORK_CLIENT_AT_CMD_RE = re.compile(
    r"(?:^|[;&|\n(]|&&|\|\|)\s*(?:[A-Za-z_]\w*=\S*\s+)*"
    r"(?:nc|ncat|netcat|telnet|socat|ssh|slogin|scp|sftp)\b"
)
# Plain openssl (dgst, enc) is local and stays out.
_OPENSSL_NETWORK_SUBCOMMANDS = frozenset({"s_client", "s_server"})
# Returns password hashes via NSS without naming a path.
_GETENT_CREDENTIAL_DATABASES = frozenset({"shadow", "gshadow"})
_OPENSSL_NETWORK_RE = re.compile(
    r"(?:^|[;&|\n(]|&&|\|\|)\s*(?:[A-Za-z_]\w*=\S*\s+)*(?:\S*/)?openssl\s+s_(?:client|server)\b"
)
# Paired with the var-executed test so `echo "${a[@]}"` is left alone.
_ARRAY_EXPANSION_RE = re.compile(r"\$\{\w+\[[@*]\]\}")
# Non-shell interpreters running an inline program (python -c, node -e, php -r): the terminal path never screens that
# program the way the python tool does. sh/bash -c are omitted, the hard-block already recursing into their payloads.
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
# Flags are per interpreter (`python -E` is not eval). Value: (exact flags, cluster letters).
_INLINE_CODE_FLAG_SPEC = {
    "python": (frozenset({"-c"}), "c"),
    "pypy": (frozenset({"-c"}), "c"),
    "node": (frozenset({"-e", "--eval"}), "e"),
    "nodejs": (frozenset({"-e", "--eval"}), "e"),
    "deno": (frozenset({"-e", "--eval"}), "e"),
    "bun": (frozenset({"-e", "--eval"}), "e"),
    "ruby": (frozenset({"-e"}), "e"),
    "perl": (frozenset({"-e", "-E"}), "eE"),
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


# -p is a print loop for perl/ruby/sed, so scoped to JS runtimes.
_NODE_PRINT_INTERPRETERS = frozenset({"node", "nodejs", "bun"})
_EVAL_SUBCOMMAND_INTERPRETERS = frozenset({"deno", "bun"})
_NODE_PRINT_FLAGS = frozenset({"-p", "--print"})
# cmd is not hard-blocked; del/erase/rd were added to the high-risk set for it.
_CMD_SHELLS = frozenset({"cmd"})
# Hard-blocked on Windows; elsewhere gate inline-command use. `pwsh script.ps1` stays out.
_POWERSHELL_INTERPRETERS = frozenset({"powershell", "pwsh"})
_VERSIONED_INTERPRETER_RE = re.compile(r"^(?:python|pypy|perl|ruby|php|node)\d+(?:\.\d+)*$")
_MULTICALL_BINARIES = frozenset({"busybox", "toybox"})
# `cd /proc/$PPID; cat environ` reads a sensitive path no single token spells.
_CHDIR_COMMANDS = frozenset({"cd", "pushd", "chdir"})
# System dirs anchored so /home/x/etc does not match; credential dotfile dirs match anywhere.
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
    """True for an interpreter whose ``-c`` / ``-e`` runs an inline program the terminal path never
    screens, including versioned python/pypy binaries."""
    return name in _INLINE_CODE_INTERPRETERS or bool(_VERSIONED_INTERPRETER_RE.match(name))


def _short_flag_cluster(token: str) -> "list[str]":
    """Split a combined short-option token into its individual flags (`-qf` -> ['-q', '-f']). A long
    option, a `-x=value` form or a bare `-` yields nothing, so only genuine clusters are
    expanded."""
    if len(token) < 3 or not token.startswith("-") or token.startswith("--") or "=" in token:
        return []
    return ["-" + ch for ch in token[1:]]


def _short_flag_arg(token: str, letters: str) -> "str | None":
    """For a short-flag cluster (``-lc``, ``-Bc``, ``-c``), if one of ``letters`` appears as a flag
    in it, return the text glued after that letter: ``""`` when the value is the next token, or
    the attached payload for ``-c'cmd'``. ``None`` when no such flag is present, or for long
    options / non-flags. Catches combined forms (``bash -lc 'git clean'``) an exact ``-c`` match
    would miss."""
    if not token.startswith("-") or token.startswith("--"):
        return None
    body = token[1:]
    for i, ch in enumerate(body):
        if ch in letters:
            return body[i + 1 :]
    return None


def _shell_quote_states(command: str) -> "list[str]":
    """The quote context of every character: ``""`` outside quoting, ``"'"`` (or ``"$'"`` for ANSI-C,
    which honours backslash escapes) inside single quoting, ``'"'`` inside double quoting, and
    ``_ESCAPED_CHAR_STATE`` for a backslash and the character it quotes. A quote mark itself reports
    the context it opens from, so a character is text bash expands exactly when its state is ``""``
    or ``'"'``.

    Tracked character by character rather than paired off with a regex, because a regex matches the
    apostrophe in `echo "it's"` against the next quote, inverting the state for everything after it.
    """
    states: "list[str]" = []
    quote = ""
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if quote in ("'", "$'"):
            # ANSI-C quoting does not protect backslashes, so `\'` there is a quote char.
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
            # Own state, so `sed "s/\$(CC)/gcc/"` is not read as a live substitution.
            states += [_ESCAPED_CHAR_STATE, _ESCAPED_CHAR_STATE]
            i += 2
            continue
        states.append(quote)
        if quote == '"':
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

    The body of a substitution is a FRESH shell context, so quoting reopens inside even when the
    whole thing sits in double quotes, and a paren the body QUOTES is text, not nesting. Counting it
    raised the depth, the real `)` then never brought it back to zero, and the span ran on past the
    end of the word, so a generated script went unnoticed.

    _shell_quote_states is a left-to-right machine, so the states it reports for a prefix are the
    ones it reports for the whole string; the window is grown until the span closes, which keeps the
    cost a constant multiple of the substitution's own length rather than a walk to the end of the
    line.
    """
    n = len(command)
    width = _SUBSTITUTION_SPAN_STEP
    while True:
        stop = min(n, start + 1 + width)
        body = command[start + 1 : stop]
        depth = 0
        for offset, state in enumerate(_shell_quote_states(body)):
            if state:
                continue
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
    """Index just past the `))` / `]` closing the arithmetic expansion at ``start``: `$((...))`, or
    the deprecated `$[...]` bash 5.2 still evaluates."""
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
    """Index just past the `}` closing the `${` at ``start``. Braces nest (`${a:-${b}}`) and a
    backslash quotes the one behind it."""
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
    """``program`` with each arithmetic expansion replaced by a digit (_ARITHMETIC_VALUE), a faithful
    stand-in because arithmetic always evaluates to an integer.

    Without it the expansion's own punctuation is read as sed source and hides the command behind
    it: `sed "$((c+1))e rm -f victim"` runs rm for real, while the raw text takes the `c` for an
    append-text command and swallows the payload as its operand. An expansion holding a COMMAND
    substitution is left alone, so the substitution stays visible to _sed_program_unresolved.
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
    """Every expansion bash performs, as the exact text each one occupies: `$(...)`, backticks,
    `${...}` in ANY form and a bare `$NAME` / `$?`.

    With ``quoted`` (the default) the text is a whole command line, so a single-quoted or
    backslash-escaped expansion is literal and reported as nothing. With ``quoted`` False the text
    is a token shlex has already unquoted, where every character counts; comparing the two tells an
    expansion the shell RUNS from one a sed program merely quotes.

    ARITHMETIC is skipped: it evaluates to an integer, so it can spell no sed command. One holding a
    command substitution is stepped INTO instead.
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
            # Skip `$` alone where the span has a nested `$(`, else the whole arithmetic span.
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
    """``text`` with each UNQUOTED newline replaced by `;`, which shlex reads as a command boundary.
    A newline inside quotes is DATA (a sed comment ends at one) so it survives, as does a
    BACKSLASH-escaped newline, which bash deletes as a line continuation rather than a separator;
    the blanket pass still supplies that boundary if one is wanted."""
    states = _shell_quote_states(text)
    out = []
    for i, ch in enumerate(text):
        if ch in "\r\n" and states[i] == "":
            if not (ch == "\n" and i and text[i - 1] == "\r"):
                out.append(";")
        else:
            out.append(ch)
    return "".join(out)


# Destructive git subcommands; reset/push/checkout only qualify with a destructive flag or pathspec.
_HIGH_RISK_GIT_SUBCOMMANDS = frozenset(
    {"clean", "restore", "rm", "update-ref", "filter-branch", "prune", "gc", "reflog"}
)
_HIGH_RISK_GIT_RESET_FLAGS = frozenset({"--hard"})
_HIGH_RISK_GIT_PUSH_FLAGS = frozenset(
    # All remote data loss, like a force push.
    {"-f", "--force", "--force-with-lease", "-d", "--delete", "--mirror", "--prune"}
)
# An unforced remove refuses on a dirty worktree.
_HIGH_RISK_GIT_WORKTREE_FLAGS = frozenset({"-f", "--force"})
_HIGH_RISK_GIT_SWITCH_FLAGS = frozenset({"-C", "-f", "--force", "--discard-changes"})
_HIGH_RISK_GIT_BRANCH_FLAGS = frozenset({"-D", "-M", "-f", "--force"})
_HIGH_RISK_GIT_STASH_ACTIONS = frozenset({"clear", "drop"})
_HIGH_RISK_GIT_CHECKOUT_FLAGS = frozenset({"-f", "--force", "-B"})
_HIGH_RISK_GIT_CHECKOUT_INDEX_FLAGS = frozenset({"-f", "--force"})
_HIGH_RISK_GIT_TAG_FLAGS = frozenset({"-d", "--delete", "-f", "--force"})
_GIT_ALIAS_ASSIGN_RE = re.compile(r"^alias\.[^=]+=(.*)$", re.DOTALL)
# The alias body comes from an env var, so the code never appears in the text.
_GIT_CONFIG_ENV_ALIAS_RE = re.compile(r"(?:^|=)alias\.", re.IGNORECASE)
_GIT_GLOBAL_VALUE_FLAGS = frozenset(
    {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path", "--config-env"}
)
# Recursively screened; the hard-block only recurses for its own smaller set.
_SHELL_C_INTERPRETERS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish", "ash"})
_COMMAND_SUBST_AT_CMD_RE = re.compile(
    r"(?:^|[;&|\n(]|&&|\|\|)\s*(?:[A-Za-z_]\w*=[^\s;&|()]*\s+)*(?:\$\(|`)"
)
# `:` is absent: its arguments are expanded but never executed. coproc's operand is a command
# position.
_SUBST_CMD_SEP = (
    r"(?:^|[;&|\n({]|&&|\|\||\b(?:then|do|else|elif|if|while|until)\b\s*"
    r"|\bcoproc\b\s*(?:[A-Za-z_]\w*\s+)?|!\s*)"
)
# Bounded, flat repetitions (nesting backtracked catastrophically). A wrapper's first plain word
# IS its command; option values are swallowed so `xargs -P $n` is not read as the command.
_ALL_WRAPPER_VALUE_FLAGS = frozenset(
    flag for flags in _WRAPPER_VALUE_FLAGS_BY_CMD.values() for flag in flags
) | {
    # Added here, not to _WRAPPER_VALUE_FLAGS_BY_CMD, which drives the high-risk classifier. `-S` is
    # absent: its operand is the argv that runs.
    "-C",
    "--chdir",
}
_WRAPPER_VALUE_FLAG_ALT = "|".join(
    re.escape(flag) for flag in sorted(_ALL_WRAPPER_VALUE_FLAGS, key = len, reverse = True)
)
# timeout accepts strtod durations, so scientific forms are valid too.
_SUBST_WRAPPER_ARG = (
    r"(?:(?:" + _WRAPPER_VALUE_FLAG_ALT + r")\s+[^\s;&|()]+"
    r"|-[^\s;&|()]*|[A-Za-z_]\w*=[^\s;&|()]*"
    r"|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[smhd]?)"
)
# Unbounded but cannot backtrack: wrapper words and args never overlap, and each repetition
# consumes mandatory whitespace. Any cap is just a count an attacker exceeds.
_SUBST_WRAPPER_RUN = (
    r"(?:(?:env|command|builtin|exec|time|nohup|nice|setsid|stdbuf|timeout|ionice|chroot"
    r"|setpriv|sudo|doas|su|xargs)\s+(?:" + _SUBST_WRAPPER_ARG + r"\s+)*)*"
)
# Redirections may precede the command word. Target and trailing space are mandatory so a
# substitution that IS the target (`>$(ls) cmd`) is not read as the command.
_SUBST_REDIR_PREFIX = r"(?:\d*(?:>>|<<<|<<|>&|<&|>|<)\s*[^\s;&|()]+\s+)*"
# `(?!\()` drops arithmetic; a double-quoted backtick opens a command word too.
_SUBST_OPENER = r"(?:\"\$\((?!\()|\$\((?!\()|\"`|`)"
_SUBST_CMD_WORD_PREFIX = (
    _SUBST_REDIR_PREFIX
    + r"(?:[A-Za-z_]\w*=[^\s;&|()]*\s+)*"
    + _SUBST_REDIR_PREFIX
    + _SUBST_WRAPPER_RUN
)
_SUBST_AT_CMD_SITE_RE = re.compile(
    _SUBST_CMD_SEP + r"\s*" + _SUBST_CMD_WORD_PREFIX + _SUBST_OPENER,
    re.IGNORECASE,
)
# A case arm's `)` is a command position, only inside case ... esac and not closing a substitution.
_SUBST_CASE_ARM_RE = re.compile(
    r"\)\s*" + _SUBST_CMD_WORD_PREFIX + _SUBST_OPENER,
    re.IGNORECASE,
)
_CASE_REGION_RE = re.compile(r"\bcase\b.*?\bin\b(.*?)(?:\besac\b|$)", re.IGNORECASE | re.DOTALL)
# The words after -exec are the executed argv.
_SUBST_EXEC_DIRECTIVE_RE = re.compile(
    r"(?:-exec(?:dir)?|--exec(?:-batch)?|-ok(?:dir)?)\s+" + _SUBST_CMD_WORD_PREFIX + _SUBST_OPENER,
    re.IGNORECASE,
)
# Lead on any separator or whitespace; widening only collects, reporting stays at command position.
_ASSIGN_LEAD = (
    r"(?:^|[\s;&|\n(){])(?:(?:export|local|readonly)\s+|(?:declare|typeset)(?:\s+[-\w+]+)*\s+)?"
)
# `(?!\()` excludes arithmetic (`sec=$((60*5)); timeout $sec make`).
_SUBST_ASSIGN_RE = re.compile(_ASSIGN_LEAD + r"([A-Za-z_]\w*)=[\"']?(?:\$\((?!\()|`)")
_PRINTF_V_ASSIGN_RE = re.compile(_ASSIGN_LEAD + r"\bprintf\s+-v\s+([A-Za-z_]\w*)\b")
# Only the value's first word counts; arrays never match.
_ASSIGN_BLOCKED_LITERAL_RE = re.compile(_ASSIGN_LEAD + r"([A-Za-z_]\w*)=([^\s;&|()]+)")


def _command_subst_body(command: str, opener: int) -> str:
    """The body text of the `$(...)` (``opener`` = index of ``(``) or backtick (``opener`` = index
    of the backtick) substitution opening there. A backtick span ends at the next unescaped
    backtick; an unterminated span of either kind runs to the end of the string."""
    if command[opener] == "`":
        closer = opener + 1
        while True:
            closer = command.find("`", closer)
            if closer < 0:
                return command[opener + 1 :]
            if command[closer - 1] != "\\":
                return command[opener + 1 : closer]
            closer += 1
    end = _substitution_span(command, opener - 1)
    return command[opener + 1 : end if end >= len(command) else end - 1]


_BLOCKED_SYNTHESIZED_COMMAND = "command substitution"
# A lookup of one literal binary (`$(which python)`) stays out.
_SUBST_ENUMERATES_COMMANDS_RE = re.compile(
    r"(?:\bcompgen\b|\b(?:ls|dir|find)\b|\b(?:echo|printf)\b[^\n;&|]*[*?[])"
)
# Sites read before refusing instead: each costs a span walk and this function has no length cap.
_MAX_SUBST_SITES = 64


@functools.lru_cache(maxsize = 4)
def _blocked_body_word_pattern_for(words: "frozenset[str]") -> "re.Pattern":
    """A blocked name appearing anywhere in a command-substitution body.

    Looser than the command-position scan on purpose: anything in the body may become the word
    that runs. `.` (the synonym for `source`) cannot share that boundary, because any dot in a
    filename then matches it - `$(cat .env)` was refused as "Blocked command(s) for safety: .".
    Punctuation names get the strict boundary, the only place they can run anyway.
    """
    word_like = sorted(w for w in words if w[:1].isalnum() or w[:1] == "_")
    punctuation = sorted(w for w in words if w not in set(word_like))
    parts = []
    if word_like:
        alt = "|".join(re.escape(w) for w in word_like)
        parts.append(rf"(?:^|[^\w./\\-])(?:[\w./\\-]*/)?({alt})(?:\.(?:exe|com|bat|cmd))?\b")
    if punctuation:
        alt = "|".join(re.escape(w) for w in punctuation)
        parts.append(rf"(?:^|[;&|`\n(])\s*({alt})(?=\s)")
    return re.compile("|".join(parts) if parts else r"(?!)")


def _subst_is_word_fragment(command: str, opener: int) -> bool:
    """Whether more of the command WORD follows the substitution opening at ``opener``.

    Bash concatenates adjacent fragments, so `$(printf r)m` and `"$(printf r)"m` both run `rm`.
    The body is benign in each, so a scan that only reads bodies reports nothing; what is
    reportable is that the executed word cannot be known.
    """
    if command[opener] == "`":
        closer = command.find("`", opener + 1)
        end = len(command) if closer < 0 else closer + 1
    else:
        end = _substitution_span(command, opener - 1)
    if end < len(command) and command[end] == '"':
        end += 1
    return end < len(command) and (command[end].isalnum() or command[end] in "_.-/")


def _case_arm_sites(
    command: str,
    quote_states: "list[str]",
    pattern: "re.Pattern | None" = None,
) -> "list[re.Match]":
    """Matches of ``pattern`` anchored on a `case` arm's `)`, which is a command position.

    Only inside `case ... in ... esac`, and only for a `)` that does not close a substitution -
    otherwise the `)` of an ordinary `echo $(date) $(ls /tmp)` would read as an arm and refuse it.
    ``pattern`` defaults to the substitution-site one; the variable-execution scan passes its own,
    since a laundered `$c` runs in an arm just as readily as a substitution does.
    """
    pattern = pattern if pattern is not None else _SUBST_CASE_ARM_RE
    if not _CASE_REGION_RE.search(command):
        return []
    closers: "set[int]" = set()
    for opener in re.finditer(r"\$\((?!\()", command):
        at = opener.start()
        if at < len(quote_states) and quote_states[at] in ("'", "$'", _ESCAPED_CHAR_STATE):
            continue
        end = _substitution_span(command, at)
        if end <= len(command):
            closers.add(end - 1)
    sites = []
    for region in _CASE_REGION_RE.finditer(command):
        start, stop = region.start(1), region.end(1)
        for site in pattern.finditer(command, start, stop):
            if site.start() not in closers:
                sites.append(site)
    return sites


def _is_wrapper_flag_operand(command: str, start: int) -> bool:
    """Whether the word at ``start`` is the VALUE of a wrapper option rather than a command.

    `xargs -P $n` gives `-P` the `$n`, which is never the command the wrapper forwards to. Checked
    here, not in the regex: a regex consuming the option and its value backtracks to the shorter
    reading, and the atomic group that would pin it needs 3.11 while this file supports 3.9.
    """
    preceding = command[:start].split()
    return bool(preceding) and preceding[-1] in _ALL_WRAPPER_VALUE_FLAGS


def _blocked_body_words(body: str) -> "set[str]":
    """Every blocked name the substitution body ``body`` (already lowercased) mentions."""
    found: "set[str]" = set()
    for groups in _blocked_body_word_pattern_for(frozenset(_BLOCKED_COMMANDS)).findall(body):
        if isinstance(groups, str):
            groups = (groups,)
        found.update(g for g in groups if g)
    return found


# Catches a substitution stashed in a variable that a later dynamic exec runs.
_HAS_COMMAND_SUBST_RE = re.compile(r"\$\((?!\()|`")
# Whole-word only: ${VENV}/bin/python still leaves a screenable basename.
_BARE_VAR_AS_COMMAND_RE = re.compile(
    r"(?:^|[;&|\n(]|&&|\|\|)\s*(?:[A-Za-z_]\w*=\S*\s+)*\$\{?\w+\}?(?=\s|$)"
)
# $VAR at command position, or a shell -c/eval payload containing `$`.
_VAR_EXECUTED_AS_COMMAND_RE = re.compile(
    r"(?:^|[;&|\n(]|&&|\|\|)\s*(?:[A-Za-z_]\w*=\S*\s+)*\$\{?\w"
    r"|\b(?:sh|bash|zsh|dash|ksh|ash)\b[^\n]*?\s-c\b[^\n]*\$"
    r"|\beval\b[^\n]*\$"
)


_SHELL_SEGMENT_SPLIT_RE = re.compile(r"^(?:;|&&|\|\||\||&)$")


_CLIENT_WRAPPERS = frozenset(
    {"env", "command", "timeout", "nohup", "nice", "ionice", "stdbuf", "setsid", "exec"}
)
_CLIENT_WRAPPER_PREFIX = (
    r"(?:(?:env|command|timeout|nohup|nice|ionice|stdbuf|setsid|exec)\s+"
    r"(?:-\S+\s+|\d+(?:\.\d+)?[smhd]?\s+)*)*"
)
# The sandbox shares the backend's env, so removing a package breaks the running process.
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
    """Tokens of the segments whose command is curl/wget, or None if there is no such segment. Keeps
    an unrelated command's option letters out of the upload scan (`ls -T && echo curl`)."""
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
        i = 0
        while i < len(seg) and re.match(r"^[A-Za-z_]\w*=", seg[i]):
            i += 1
        if i >= len(seg):
            continue
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
    """curl/wget used to run remote code (piped into a shell, or via process substitution) or to
    upload local data. Plain downloads stay out. Fails closed on an unparseable command."""
    low = command.lower()
    if _NETWORK_CLIENT_AT_CMD_RE.search(command) or _OPENSSL_NETWORK_RE.search(low):
        return True
    # An argument-position mention is not an invocation.
    has_curl = bool(_CURL_AT_CMD_RE.search(command))
    has_wget = bool(_WGET_AT_CMD_RE.search(command))
    if not has_curl and not has_wget:
        return False
    if _PIPE_TO_INTERPRETER_RE.search(low):
        return True
    if "<(" in command:
        return True
    try:
        tokens = shlex.split(command.replace("\n", " "), posix = True)
    except ValueError:
        return True
    # Scope flags to the segment running curl/wget.
    tokens = _tokens_for_client_segment(tokens, has_curl, has_wget)
    if tokens is None:
        return False
    method_pending = False
    for t in tokens:
        name = t.split("=", 1)[0]
        # Separated, attached (-XDELETE) and --request= forms.
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
            or (not name.startswith("--") and name.startswith(_CURL_UPLOAD_SHORT_FLAGS))
        ):
            return True
        if has_wget and name in _WGET_UPLOAD_FLAGS:
            return True
    return False


_GIT_CLEAN_DRY_RUN_FLAGS = frozenset({"-n", "--dry-run"})


def _container_subcommand_is_read_only(tokens: list, start: int) -> bool:
    """Whether a container CLI's first positional is a read subcommand. A bare `docker` or `docker
    --version` prints help and runs nothing."""
    for t in tokens[start + 1 :]:
        if t in _SHELL_SEPARATORS or not set(t) - set(";&|()"):
            break
        if t.startswith("-"):
            continue
        return t.lower() in _CONTAINER_READ_SUBCOMMANDS
    return True


def _segment_has_command_after(tokens: list, start: int) -> bool:
    """Whether a command word follows an assignment in the same segment. A bare `export PATH=...` or
    `FOO=bar` runs nothing: every terminal call gets its own shell process, so an assignment with
    no command dies with it."""
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
    """Whether a flag appears in the same command segment as ``start``, so a later command's options
    are not read as this command's."""
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
    """Whether a recursive flag (-R / --recursive / an -rf style cluster) belongs to the command
    starting at ``start``: scan only up to the next separator, so `grep -R x . && chmod +x f`
    does not make the chmod look recursive."""
    for t in tokens[start + 1 :]:
        if t in _SHELL_SEPARATORS or not set(t) - set(";&|()"):
            break
        if t in ("-R", "--recursive"):
            return True
        if t[:1] == "-" and t[:2] != "--" and "=" not in t and "R" in t[1:]:
            return True
    return False


def _inline_python_is_high_risk(code: str) -> bool:
    """Screen a `python -c` payload with the same analyzer the python tool uses, so an ordinary
    one-liner runs and a destructive one still asks. Source that does not parse fails closed:
    shell quoting may have mangled it, leaving nothing to screen."""
    if _parse_python(code)[1] is not None:
        return True
    return _python_is_high_risk(code)


def _terminal_is_high_risk(command: str, _depth: int = 0) -> bool:
    """High-risk terminal command for auto mode: credential/secret access, privilege escalation,
    destructive/persistence changes, or network exec/exfil. Ordinary dev commands run without a
    prompt. Fails closed (prompts) on an unparseable command. ``_depth`` bounds the recursion
    into shell ``-c`` payloads."""
    if len(command) > _MAX_TERMINAL_SCAN_CHARS:
        # Far longer than any ordinary command, and screening it is superlinear, so it asks instead.
        return True
    if not command or not command.strip():
        return False
    if (
        _depth == 0
        and _reads_differently_under_cmd(command)
        and _terminal_is_high_risk(_cmd_reading(command))
    ):
        return True
    if _command_references_sensitive(command):
        return True
    # A bare redirection truncates the file, like `truncate -s 0`.
    if _BARE_TRUNCATING_REDIRECT_RE.search(command):
        return True
    if _PROC_SUBST_EXEC_RE.search(command):
        return True
    if _PKG_REMOVE_AT_CMD_RE.search(command):
        return True
    if _PIPE_TO_INTERPRETER_RE.search(command.lower()):
        return True
    _herestring = _HERESTRING_TO_INTERPRETER_RE.search(command)
    if _herestring:
        return True
    # shlex reads newlines as whitespace; ANSI-C quoting hides names.
    decoded = _decode_ansi_c(command, keep_one_word = True)
    normalized = decoded.replace("\r\n", ";").replace("\n", ";").replace("\r", ";")
    # Identical unless a newline is present, so single-line commands skip the quote walk.
    quoted_newlines_kept = (
        _separate_unquoted_newlines(decoded) if "\n" in decoded or "\r" in decoded else normalized
    )
    # Tells expansions the shell runs from ones a sed program quotes; held in both newline forms.
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
    expanded = _expand_shell_assignments(_expand_param_defaults(normalized))
    # Also over the expanded form so variable-assembled curl/wget names are seen.
    if _command_is_network_exec_or_exfil(command) or _command_is_network_exec_or_exfil(expanded):
        return True
    if _COMMAND_SUBST_AT_CMD_RE.search(command):
        return True
    # Plain assignments were expanded above, so this binding is unfollowable: fail closed.
    if _HAS_COMMAND_SUBST_RE.search(command) and _VAR_EXECUTED_AS_COMMAND_RE.search(command):
        return True
    if _BARE_VAR_AS_COMMAND_RE.search(expanded):
        return True
    # Arrays are not resolved by assignment expansion, so the check above misses them.
    if _ARRAY_EXPANSION_RE.search(command) and _VAR_EXECUTED_AS_COMMAND_RE.search(command):
        return True
    # Also scan a pass that splits only unquoted newlines: a quoted newline is data (it ends a
    # sed comment). Adds detections without merging commands.
    for text in {normalized, expanded, quoted_newlines_kept}:
        try:
            lexer = shlex.shlex(text, posix = True, punctuation_chars = ";&|()")
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            return True
        # Per expansion pass, so variable-assembled paths are judged resolved.
        if _terminal_reaches_outside_sandbox(tokens, text):
            return True
        recursive = any(
            t in ("-R", "--recursive")
            or (t[:1] == "-" and t[:2] != "--" and "=" not in t and "R" in t[1:])
            for t in tokens
        )
        find_like = any(_token_command_base(t) in ("find", "fd") for t in tokens)
        sed_scan_limit = _sed_scan_limit(
            sum(1 for t in tokens if _token_command_base(t) in _SED_COMMANDS)
        )
        sed_vars: "dict[str, str] | None" = None
        sed_bindings: "list[tuple[int, str, str | None]] | None" = None
        sed_cursor = 0
        # Built lazily, once per pass, only when a sed is reached.
        sed_stops: "frozenset[int] | None" = None
        sed_skips: "frozenset[int]" = frozenset()
        sed_quoted: "frozenset[int]" = frozenset()
        sed_globs: "frozenset[int]" = frozenset()
        sed_expandable: "frozenset[int]" = frozenset()
        if find_like and any(t.split("=", 1)[0] in _HIGH_RISK_FIND_FLAGS for t in tokens):
            return True
        if any(_token_command_base(t) in _ARG_EXEC_FLAG_OWNERS for t in tokens) and any(
            t.split("=", 1)[0] in _HIGH_RISK_ARG_EXEC_FLAGS for t in tokens
        ):
            return True
        if _LISTENER_PY_MODULE_RE.search(text) or _LISTENER_BIN_AT_CMD_RE.search(text):
            return True
        expect_command = True
        prefix_pending = False
        scan_forward = False
        current_command = ""
        git_subcommand = ""
        shell_c_pending = False
        wrapper_value_pending = False
        exec_flag_pending = False
        git_checkout_positionals = 0
        git_worktree_action = ""
        win_operand_pending = False
        inline_python_pending = False
        py_module_pending = False
        git_submodule_action = ""
        awk_program_pending = False
        git_config_alias_pending = False
        git_glob_pending = False
        chdir_pending = False
        xargs_index = -1
        coproc_kw = False
        for _tok_idx, token in enumerate(tokens):
            after_coproc = coproc_kw
            coproc_kw = False
            if (
                token in _SHELL_SEPARATORS
                or (token in _SHELL_KEYWORDS_AS_SEP and expect_command)
                # Wrapper command position lives in `prefix_pending`, so `time coproc rm` is caught here.
                or (token == "coproc" and prefix_pending)
                or not set(token) - set(";&|()")
            ):
                coproc_kw = (expect_command or prefix_pending) and token == "coproc"
                expect_command = True
                prefix_pending = False
                xargs_index = -1
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
                # The operand sits where the command word would be.
                win_operand_pending = token.lower() != "not"
                continue
            if win_operand_pending:
                win_operand_pending = False
                continue
            if expect_command and _REDIR_PREFIX_RE.match(token):
                continue
            if exec_flag_pending and token == "--":
                # Nothing after fd's `--` is an option.
                exec_flag_pending = False
                continue
            if token.startswith("-"):
                flag = token.split("=", 1)[0]
                if exec_flag_pending and flag in _EXEC_FORWARD_FLAGS:
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
                    # fd takes the command attached to the short option too (`-xrm`, fdfind 9.0.0).
                    attached = token[2:].strip("\"'")
                    if attached and (_depth >= 3 or _terminal_is_high_risk(attached, _depth + 1)):
                        return True
                    scan_forward = True
                    expect_command = True
                    continue
                if current_command == "setpriv" and flag in _SETPRIV_PRIVILEGE_FLAGS:
                    # Ahead of the wrapper-value skip, which would swallow `--reuid 0`.
                    return True
                if (
                    prefix_pending
                    and "=" not in token
                    and flag in _WRAPPER_VALUE_FLAGS_BY_CMD.get(current_command, frozenset())
                ):
                    wrapper_value_pending = True
                    continue
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
                    # Python payloads go through the python analyzer; other runtimes stay gated.
                    if _current_is_python_family:
                        # A bare `-c` yields "", not None; only a non-empty value is attached.
                        _attached = _short_flag_arg(token, _inline_spec[1])
                        if _attached:
                            if _depth >= 3 or _inline_python_is_high_risk(_attached):
                                return True
                            continue
                        inline_python_pending = True
                        continue
                    return True
                if current_command in _NODE_PRINT_INTERPRETERS and (
                    flag in _NODE_PRINT_FLAGS or _short_flag_arg(token, "p") is not None
                ):
                    return True
                if current_command in _POWERSHELL_INTERPRETERS and flag.lower().startswith(
                    ("-c", "-e")
                ):
                    return True
                if current_command in _SHELL_C_INTERPRETERS:
                    payload = _short_flag_arg(token, "c")
                    if payload is not None:
                        # Plain letters after `c` (bash -ce) are more options, not an attached payload.
                        if payload and payload.isalpha() and len(payload) <= 4:
                            shell_c_pending = True
                        elif payload:
                            if _depth >= 3:
                                return True
                            if _terminal_is_high_risk(payload, _depth + 1):
                                return True
                        else:
                            shell_c_pending = True
                # env -S runs a string as a command; env -C chdirs (enabling relative sensitive reads).
                if current_command == "env":
                    if flag in ("-C", "--chdir"):
                        return True
                    payload = None
                    if token.startswith("-S") and token != "-S":
                        payload = token[2:]
                    elif flag == "--split-string" and "=" in token:
                        payload = token.split("=", 1)[1]
                    elif token == "-S" or flag == "--split-string":
                        shell_c_pending = True
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
                    if git_subcommand == "reset" and flag in _HIGH_RISK_GIT_RESET_FLAGS:
                        return True
                    if git_subcommand == "push" and (
                        flag in _HIGH_RISK_GIT_PUSH_FLAGS
                        or any(f in _HIGH_RISK_GIT_PUSH_FLAGS for f in _short_flag_cluster(token))
                    ):
                        return True
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
                    if git_subcommand == "branch" and (
                        flag in _HIGH_RISK_GIT_BRANCH_FLAGS
                        or any(f in _HIGH_RISK_GIT_BRANCH_FLAGS for f in _short_flag_cluster(token))
                    ):
                        return True
                    # The value comes from the environment, so an alias key would store unscreened code.
                    if flag == "--config-env" and _GIT_CONFIG_ENV_ALIAS_RE.search(token):
                        return True
                    if not git_subcommand and "=" not in token and flag in _GIT_GLOBAL_VALUE_FLAGS:
                        git_glob_pending = True
                continue
            if _ASSIGNMENT_RE.match(token):
                _assign_name, _, _assign_value = token.partition("=")
                if current_command == "alias" and _assign_value:
                    if _depth >= 3 or _terminal_is_high_risk(_assign_value, _depth + 1):
                        return True
                # Only for the command they prefix: a bare `export PATH=...` runs nothing.
                if _env_assignment_is_unsafe(
                    _assign_name, _assign_value
                ) and _segment_has_command_after(tokens, _tok_idx):
                    return True
                continue
            raw = token.strip(";&|()`{}")
            if not raw:
                continue
            # /c is not a `-`-flag, so it is handled in argument position.
            if current_command in _CMD_SHELLS and raw.lower() in ("/c", "/k"):
                shell_c_pending = True
                continue
            if shell_c_pending:
                shell_c_pending = False
                # An unquoted payload spans the remaining tokens.
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
            if git_glob_pending:
                git_glob_pending = False
                # An alias body is code git runs later: screen it.
                m = _GIT_ALIAS_ASSIGN_RE.match(raw)
                if m and _depth < 3:
                    alias_body = m.group(1)
                    # A `!` alias runs through a shell; otherwise screen it as `git <body>`.
                    nested = alias_body[1:] if alias_body.startswith("!") else "git " + alias_body
                    if _terminal_is_high_risk(nested, _depth + 1):
                        return True
                continue
            if wrapper_value_pending:
                wrapper_value_pending = False
                continue
            if prefix_pending and _WRAPPER_DURATION_RE.fullmatch(raw):
                continue
            base = os.path.basename(raw).lower()
            stem, ext = os.path.splitext(base)
            if ext in {".exe", ".com", ".bat", ".cmd"}:
                base = stem
            coproc_name_here = after_coproc and _is_coproc_name(tokens, _tok_idx)
            if (
                (expect_command or prefix_pending)
                # A coprocess NAME is not a wrapper (`coproc env if git clean -fd; then :; fi` deletes).
                and not coproc_name_here
                and (
                    base in _AUTO_SAFE_WRAPPERS
                    or base in _MULTICALL_BINARIES
                    or base in _PRIVILEGE_EXEC_WRAPPERS
                )
            ):
                # Track the wrapper so its own flags (env -S / -C) are judged meanwhile.
                prefix_pending = True
                expect_command = False
                current_command = base
                continue
            if expect_command or prefix_pending or scan_forward:
                if base in _HIGH_RISK_COMMANDS or base.startswith("mkfs"):
                    # Container CLIs: only read subcommands (ps, logs) stay out.
                    if not (
                        base in _CONTAINER_CLIS
                        and _container_subcommand_is_read_only(tokens, _tok_idx)
                    ):
                        return True
                # bash expands a command-position glob later (`/bin/r[m]`), so the name is unknown: ask.
                if _is_unresolved_command_glob(base):
                    return True
                if base in _LISTENER_BINARIES:
                    return True
                if base in _HIGH_RISK_RECURSIVE_COMMANDS and _segment_is_recursive(
                    tokens, _tok_idx
                ):
                    return True
                if base in _HIGH_RISK_FORWARDING_COMMANDS:
                    if base == "xargs" and xargs_index < 0:
                        # xargs builds the argv of what follows, so a sed there may get an unseen program.
                        xargs_index = _tok_idx
                    # find/fd run a child only at -exec/-ok, so `find . -name rm` stays quiet.
                    if base in _EXEC_FLAG_FORWARDING_COMMANDS:
                        scan_forward = False
                        exec_flag_pending = True
                    else:
                        scan_forward = True
                elif base == "git":
                    # Only git stops forwarding: its risk is the subcommand; find's predicates precede -exec.
                    scan_forward = False
                current_command = base
                if base in _CHDIR_COMMANDS:
                    chdir_pending = True
                if base in _AWK_COMMANDS:
                    awk_program_pending = True
                if base in _SED_COMMANDS:
                    # `e` / `s///e` may ride on -e rather than the next positional.
                    if sed_stops is None:
                        # A quoted `';'` operand is a sed FILE, not the end of the invocation; redirections never
                        # reach sed.
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
                        # Script pushed past the scan window: not looked at, so ask.
                        return True
                    if _sed_program_is_a_placeholder(sed_program):
                        return True
                    if xargs_index >= 0 and _xargs_hides_sed_program(
                        tokens, xargs_index, _tok_idx, sed_program
                    ):
                        # xargs builds argv from stdin, so the program is not in the text.
                        return True
                    if "$" in sed_program:
                        # Resolve variable programs in this pass only: it keeps the quoted newline ending a comment.
                        # Only earlier assignments count, the last wins.
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
                    # An unresolved program the shell still builds asks, but only where its own occurrence is
                    # expanded, so `echo "$p"; sed 's/$p/x/' f` stays quiet.
                    if sed_live and _sed_program_unresolved(sed_variants, live_expansions):
                        return True
            elif current_command == "git" and not git_subcommand:
                git_subcommand = base
                if base == "clean" and _segment_has_flag(
                    tokens, _tok_idx, _GIT_CLEAN_DRY_RUN_FLAGS, "n"
                ):
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
                # `git submodule foreach '<cmd>'` runs its argument.
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
                return True
            elif current_command == "openssl" and base in _OPENSSL_NETWORK_SUBCOMMANDS:
                # The command-position regex misses wrapped forms.
                return True
            elif current_command == "sysctl" and "=" in raw:
                return True
            elif (
                current_command == "git"
                and git_subcommand == "worktree"
                and not git_worktree_action
            ):
                git_worktree_action = base
            elif current_command in _EVAL_SUBCOMMAND_INTERPRETERS and base == "eval":
                return True
            elif current_command == "git" and git_subcommand == "checkout" and base == ".":
                return True
            elif current_command == "git" and git_subcommand == "checkout":
                # A second positional is a pathspec that overwrites the file; one is ambiguous with a branch.
                git_checkout_positionals += 1
                if git_checkout_positionals >= 2:
                    return True
            elif (
                current_command == "git" and git_subcommand == "config" and git_config_alias_pending
            ):
                git_config_alias_pending = False
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
                return True
            elif current_command == "git" and git_subcommand == "push" and raw[:1] in ("+", ":"):
                # `+src:dst` / `:dst` refspecs are the punctuation form of --force / --delete.
                if len(raw) > 1:
                    return True
            elif chdir_pending:
                chdir_pending = False
                if any(
                    _SENSITIVE_CHDIR_RE.search(cand)
                    for cand in (raw, _expand_param_defaults(raw), _expand_shell_assignments(raw))
                ):
                    return True
            expect_command = coproc_name_here
            prefix_pending = False
    return False


def _python_is_high_risk(code: str) -> bool:
    """High-risk python for auto mode: code the sandbox static analysis would refuse anyway (shell
    escape, network egress, a sensitive read), that reads/writes a credential path, or that runs
    dynamically built code past those static checks. Ordinary in-workdir file writes and
    computation run without a prompt."""
    if not code or not code.strip():
        return False
    # A confirmation beats a silent refusal at execution time.
    if _check_code_safety(code) is not None:
        return True
    tree, _parse_error = _parse_python(code)
    if _parse_error is not None:
        return _references_sensitive_path(code)
    # The workdir confines only relative paths.
    if _python_reaches_outside_sandbox(tree, code):
        return True
    # Credential basenames only count inside strings: `credentials = {}` does no I/O.
    for _node in _tree_nodes(tree):
        if (
            isinstance(_node, ast.Constant)
            and isinstance(_node.value, str)
            and _references_sensitive_path(_node.value)
        ):
            return True
    # Parity with the terminal `rm` gate.
    destructive_fs_aliases: "set[str]" = set()
    # So an unrelated .kill() on a user object is not mistaken for one.
    psutil_names: "set[str]" = set()
    for _node in _tree_nodes(tree):
        if isinstance(_node, ast.Import):
            for _a in _node.names:
                if _a.name.split(".")[0] in _PY_PROCESS_MODULES:
                    psutil_names.add("psutil")
        elif (
            isinstance(_node, ast.ImportFrom)
            and (_node.module or "").split(".")[0] in _PY_PROCESS_MODULES
        ):
            psutil_names.add("psutil")
    os_module_aliases: "set[str]" = {"os", "posix", "nt"}

    def _is_os_module_ref(value) -> bool:
        # builtins.__import__ is the same callable reached through the module.
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

    for node in _tree_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _PY_DESTRUCTIVE_FS_MODULES:
            for alias in node.names:
                if alias.name in _PY_DESTRUCTIVE_FS_IMPORT_NAMES:
                    destructive_fs_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("os", "posix", "nt") and alias.asname:
                    os_module_aliases.add(alias.asname)
        elif isinstance(node, ast.Assign) and _is_os_module_ref(node.value):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    os_module_aliases.add(tgt.id)
        elif isinstance(node, ast.NamedExpr) and _is_os_module_ref(node.value):
            if isinstance(node.target, ast.Name):
                os_module_aliases.add(node.target.id)

    def _is_fs_module_ref(value) -> bool:
        if _is_os_module_ref(value):
            return True
        return isinstance(value, ast.Name) and value.id in _PY_DESTRUCTIVE_FS_MODULES

    def _is_process_kill(node) -> bool:
        if "psutil" not in psutil_names:
            return False
        return isinstance(node, ast.Attribute) and node.attr in _PY_PROCESS_KILL_ATTRS

    def _is_destructive_attr(attr: str, value) -> bool:
        if attr in _PY_DESTRUCTIVE_FS_ATTRS:
            return True
        return attr in _PY_DESTRUCTIVE_FS_OS_ATTRS and _is_os_module_ref(value)

    def _module_dict_target(value):
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
        # Namespace-dict lookup is getattr spelled differently; anchored to fs modules.
        if not isinstance(node, ast.Subscript):
            return False
        module = _module_dict_target(node.value)
        if module is None:
            return False
        attr = _folded_str_literal(node.slice)
        if attr is None:
            return _is_fs_module_ref(module)
        return _is_destructive_attr(attr, module)

    # A stored getattr lookup is called later, so bind the name here.
    for node in _tree_nodes(tree):
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

    # Gated via the file handle: pandas DataFrame.truncate() is common and harmless.
    file_handles: "set[str]" = set()
    for node in _tree_nodes(tree):
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
        for node in _tree_nodes(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "truncate"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in file_handles
            ):
                return True
    # A bound reference hides the call site behind a plain Name.
    for node in _tree_nodes(tree):
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
            if _is_destructive_attr(node.value.attr, node.value.value):
                destructive_fs_aliases.add(node.target.id)
    for node in _tree_nodes(tree):
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
            inner = func.value
            if isinstance(inner, ast.Attribute) and _is_destructive_attr(inner.attr, inner.value):
                return True
            if isinstance(inner, ast.Name) and inner.id in destructive_fs_aliases:
                return True
            if _is_module_dict_lookup(inner):
                return True
        # Fold the name ("un" + "link"); an unfoldable name on an fs module fails closed.
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
    # Fold split paths through variables; unresolved fragments use a sentinel so partial folds
    # never false-positive.
    str_vars: "dict[str, str]" = {}
    for node in _tree_nodes(tree):
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
            folded = _folded_path(value, str_vars)
            if folded and "\x00" not in folded and "\x02" not in folded:
                str_vars[node.targets[0].id] = folded

    for node in _tree_nodes(tree):
        if isinstance(node, (ast.BinOp, ast.JoinedStr, ast.Call)):
            folded = _folded_path(node, str_vars)
            if folded and _folded_is_sensitive(folded):
                return True
    # Non-literal exec/eval/compile/__import__ runs unseen code: ask. Literal ones are screened.
    for node in _tree_nodes(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            if func.attr == "import_module":
                name = "__import__"
            elif func.attr in ("exec", "eval", "compile"):
                name = func.attr
        if name not in ("exec", "eval", "compile", "__import__"):
            continue
        arg = node.args[0] if node.args else None
        if arg is None:
            for kw in node.keywords:
                if kw.arg in ("source", "name"):
                    arg = kw.value
                    break
        if arg is None:
            continue
        if isinstance(arg, ast.Constant) and isinstance(arg.value, (str, bytes)):
            if name == "__import__":
                # A literal __import__("socket") gets the same module screen as a static import.
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
    """Whether a tool call is sensitive enough to pause for approval in auto (Approve for me) mode.

    Unlike is_potentially_unsafe_tool_call (which prompts on anything not read-only), this prompts
    only on genuinely sensitive actions - credential access, privilege escalation,
    destructive/persistence changes, and network exec/exfil - and lets ordinary development commands
    run. The hard-block command set, rlimits and secret-env stripping remain in force underneath.
    Unknown tools fail closed (prompt).
    """
    if _web_search_fetches_url(name, arguments):
        return True
    if name in _ALWAYS_SAFE_TOOLS:
        return False
    if name == "render_html":
        return _render_html_reaches_network(arguments)
    if name.startswith(MCP_TOOL_PREFIX):
        tool_name = _mcp_raw_tool_name(name)
        if tool_name in _BLENDER_CLI_SUMMARY_TOOLS:
            return True
        # One vocabulary for the term-boundary regexes.
        tool_name = _CAMEL_CASE_RE.sub("_", _MCP_TERM_SEPARATOR_RE.sub("_", tool_name))
        # Exec tools, credential nouns and sensitive paths prompt; ordinary CRUD calls run.
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
        # A read-named tool carrying a destructive payload masks a destructive action.
        if _mcp_arguments_mutate(arguments):
            return True
        # MCP names are open vocabulary, so a name with no recognised verb asks.
        if not _mcp_verb_is_known(tool_name):
            return True
        return False
    if name == "terminal":
        return _terminal_is_high_risk(str(arguments.get("command", "")))
    if name == "python":
        return _python_is_high_risk(str(arguments.get("code", "")))
    return True


def _canon_win_path(p: str) -> str:
    """Canonical form for trust comparison: realpath (expands 8.3 aliases and resolves
    junctions/symlinks) + normcase/normpath."""
    return os.path.normcase(os.path.normpath(os.path.realpath(p)))


def _augment_native_program_roots(roots: list[str]) -> list[str]:
    """Add the native Program Files sibling for any x86 root by stripping the `` (x86)`` suffix, so
    a 32-bit process (whose known-folder ids map only to the x86 root) still trusts a 64-bit Git
    install."""
    out = list(roots)
    for root in roots:
        base = root.rstrip("\\/")
        if base.lower().endswith(" (x86)"):
            native = base[: -len(" (x86)")]
            if native and native not in out:
                out.append(native)
    return out


def _windows_program_roots() -> list[str]:
    """Program Files install roots, resolved ONLY from the Windows known-folder API
    (SHGetKnownFolderPath). Fails closed (returns ``[]``) if the API is unavailable: env vars
    (%ProgramFiles%, even %SystemDrive%) are caller-overrideable and could relocate the trust
    boundary. On any real Windows host shell32 is present, so this only returns empty where the
    sandbox git-PATH feature is not needed anyway (#7317)."""
    roots: list[str] = []
    try:
        import ctypes
        from ctypes import wintypes

        # The X64 id (Win10 1703+) yields the native root even from a 32-bit process.
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
    """Find a git launcher in a TRUSTED Program Files dir. Returns ``(canonical_dir, ext)`` or ``("",
    "")``.

    ``shutil.which`` returns only the first PATH match, which may be an untrusted user shim; scan
    the remaining PATH entries for a later trusted Git so bare ``git`` still resolves (#7317).
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

    Only the Program Files roots are trusted (admin-writable only), resolved via the known-folder
    API so an overridden env var cannot relocate them, never ``%SystemRoot%`` (Git does not install
    there and it holds world-writable subdirs like ``Windows\\Temp``). Per-user managers
    (Scoop/Choco shims under the profile) are refused. Paths are canonicalized so 8.3 aliases and
    junctions still resolve to their real root (#7317).
    """
    norm = _canon_win_path(path)
    for root in _windows_program_roots():
        root_norm = _canon_win_path(root)
        if norm == root_norm or norm.startswith(root_norm + os.sep):
            return True
    return False


# Not dot-named (walks skip dot-dirs) and not "tmp" (too common). Shared with os_sandbox: a
# drifted name would silently disable the scan's exemption.
_SANDBOX_TEMP_DIRNAME = os_sandbox.TOOL_TEMP_DIRNAME


def _sandbox_temp_dir(workdir: str) -> str:
    """The scratch directory for a sandboxed child, created when missing.

    Inside the workdir, not the workdir itself: Git for Windows mounts /tmp at %TEMP% (the msys2
    ``usertemp`` fstab entry), so pointing TEMP at the workdir made /tmp its shortest POSIX name and
    ``pwd`` printed /tmp, leaving the user no way to find the real folder (#8892). One level down
    still sits where the listings reach.

    Falls back to the workdir when the name is unusable, since a TMPDIR that does not exist breaks
    every tempfile call in the child. os.mkdir, never os.makedirs, so a workdir deleted mid-call is
    not silently recreated.
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

    Exact stored spelling, because the walks read the name off os.walk: on a case-insensitive volume
    (default APFS, every NTFS) our lowercase probe lands on a directory stored as ``TMP``, and
    realpath does not canonicalise case. And the real directory, not a link or junction to one,
    since os.walk does not follow links and the artifacts would land where both walks skip.
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

    Identity plus writability, since tempfile abandons an unwritable TMPDIR for the platform
    default. Path accounting asks _is_sandbox_temp_dir instead: which directory a segment sits in
    does not depend on writability.
    """
    return _is_sandbox_temp_dir(temp_dir, workdir) and os.access(temp_dir, os.W_OK | os.X_OK)


# Git for Windows' MSYS sh.exe cannot start inside MXC, so disable hooks/pager/editor.
# Compatibility defaults, not a boundary.
_ISOLATED_CMD_GIT_ENV = {
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "core.hooksPath",
    "GIT_CONFIG_VALUE_0": "NUL",
    "GIT_PAGER": "",
    "GIT_EDITOR": "unsloth-no-editor",
    "GIT_SEQUENCE_EDITOR": "unsloth-no-editor",
    "GIT_TERMINAL_PROMPT": "0",
    "GCM_INTERACTIVE": "never",
}


def _build_safe_env(workdir: str, shell: "str | None" = None) -> dict[str, str]:
    """Build a minimal, credential-free environment for sandboxed subprocesses.

    Whitelist-built from scratch (parent env NOT inherited): only
    PATH/HOME/TMPDIR/LANG/TERM/PYTHONIOENCODING/PYTHONPATH (+VIRTUAL_ENV or Windows SystemRoot and a
    minimal PATHEXT) reach the child; all credential vars (HF_TOKEN, AWS_*, etc.) are absent. HOME
    (and on Windows HOMEDRIVE/HOMEPATH) points at the sandbox workdir so SDKs can't read the
    operator's cached creds, and the temp vars at _sandbox_temp_dir just inside it. PYTHONPATH
    carries only the sandbox sitecustomize shim directory.

    PATH starts with the Unsloth interpreter / venv and OS system dirs so ``python``/``pip`` stay
    pinned. On Windows only, Git-for-Windows install dirs from the host PATH are appended so bare
    ``git`` resolves (#7317). User-writable host PATH entries are never inherited: they could shadow
    auto-safe terminal commands.

    ``shell="cmd_isolated"`` is the Windows MXC Terminal on cmd.exe: Git Bash's userland stays off
    PATH and git gets the non-interactive settings in _ISOLATED_CMD_GIT_ENV.
    """
    exe_dir = os.path.dirname(sys.executable)
    path_entries = [exe_dir] if exe_dir else []

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        venv_bin = os.path.join(venv, "Scripts" if sys.platform == "win32" else "bin")
        if venv_bin not in path_entries:
            path_entries.append(venv_bin)

    if sys.platform == "win32":
        sysroot = os.environ.get("SystemRoot", r"C:\Windows")
        # Ahead of System32 (bare `find` would hit FIND.EXE), behind the interpreter dirs so Git's
        # python.exe cannot shadow this environment.
        if shell != "cmd_isolated":
            path_entries.extend(_windows_bash_userland_dirs())
        path_entries.extend([os.path.join(sysroot, "System32"), sysroot])
    else:
        path_entries.extend(["/usr/local/bin", "/usr/bin", "/bin"])

    # Inherit the host git's dir only under a system install root; user-writable shim dirs would
    # let a planted rg.exe run as an auto-approved command.
    git_ext = ""
    if sys.platform == "win32":
        # Canonical path, scanning past untrusted shims; cannot be retargeted via a junction later.
        _trusted_git_dir, git_ext = _resolve_trusted_windows_git()
        if not _trusted_git_dir and shell == "cmd_isolated":
            _trusted_git_dir, git_ext = _windows_bash_git_cmd_dir(), ".EXE"
        if _trusted_git_dir:
            path_entries.append(_trusted_git_dir)

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
        # Remaps ChatGPT code-interpreter paths (/mnt/data) onto the sandbox cwd.
        "PYTHONPATH": _SANDBOX_SITE_DIR,
    }
    if venv:
        env["VIRTUAL_ENV"] = venv
    # Windows needs SystemRoot for Python/subprocess to work.
    if sys.platform == "win32":
        env["SystemRoot"] = os.environ.get("SystemRoot", r"C:\Windows")
        # Windows honours TEMP/TMP, not TMPDIR.
        env["TEMP"] = temp_dir
        env["TMP"] = temp_dir
        # Path.home() ignores HOME on Windows; a workdir USERPROFILE would instead send pip's cache to .\pip in the cwd.
        env["HOMEDRIVE"], env["HOMEPATH"] = os.path.splitdrive(workdir)
        # Restrict PATHEXT so cwd .BAT/.CMD cannot hijack bare names.
        pathext = ".EXE;.COM"
        if git_ext and git_ext not in (".EXE", ".COM"):
            pathext += ";" + git_ext
        env["PATHEXT"] = pathext
        # Bare names otherwise search cwd before PATH on Windows.
        env["NoDefaultCurrentDirectoryInExePath"] = "1"
        if shell == "cmd_isolated":
            env.update(_ISOLATED_CMD_GIT_ENV)
    return env


# Dropped even in bypass mode; over-strips on purpose.
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
        # Auth brokers / capability handles, listed by name. Credentialed URL values are dropped by
        # _is_secret_env_value() regardless of name.
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
    "AUTH",
    "CONNSTR",
    "CONNECTIONSTRING",
)
# Hardening flags that look secret but must be kept (AWS_EC2_METADATA_DISABLED).
_BYPASS_ENV_KEEP_NAMES = frozenset(
    {
        "AWS_EC2_METADATA_DISABLED",
        "AWS_EC2_METADATA_V1_DISABLED",
    }
)
# Userinfo must precede the first '/', so an '@' in a path does not match.
_URL_USERINFO_RE = re.compile(r"://[^/\s@]+@")
# Connection-string credential fields; `...Name=` fields do not match.
_SECRET_VALUE_RE = re.compile(r"(?i)(?:password|pwd|accountkey|accesskey)\s*=\s*[^\s;]")

# Pointers to real cred files that would defeat the HOME repoint.
_BYPASS_ENV_CRED_LOCATION_NAMES = frozenset(
    {
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_XET_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "HF_ASSETS_CACHE",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
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
        "NPM_CONFIG_USERCONFIG",
        "NPM_CONFIG_GLOBALCONFIG",
        "YARN_RC_FILENAME",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "CARGO_HOME",
        "RCLONE_CONFIG",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "BASH_ENV",
        "HOMEDRIVE",
        "HOMEPATH",
    }
)
# Windows profile dirs SDKs read creds under; repointed (not dropped) since callers expect them present.
_BYPASS_ENV_WINDOWS_PROFILE_VARS = ("USERPROFILE", "APPDATA", "LOCALAPPDATA")


def _is_secret_env_name(name: str) -> bool:
    """True if an env var name looks like it carries a credential."""
    upper = name.upper()
    if upper in _BYPASS_ENV_KEEP_NAMES:
        return False
    if upper in _BYPASS_ENV_SECRET_NAMES:
        return True
    if any(upper.startswith(p) for p in _BYPASS_ENV_SECRET_PREFIXES):
        return True
    return any(marker in upper for marker in _BYPASS_ENV_SECRET_MARKERS)


def _is_cred_location_env_name(name: str) -> bool:
    """True for vars that point SDKs at the real home/cache/config (cached creds)."""
    return name.upper() in _BYPASS_ENV_CRED_LOCATION_NAMES


def _is_secret_env_value(value: str) -> bool:
    """True if a value embeds credentials regardless of its name: URL userinfo
    (``scheme://user:token@host`` in DATABASE_URL / PIP_INDEX_URL / HTTP_PROXY) and
    connection-string credential fields (``...;Password=...`` / ``...;AccountKey=...``) whose
    names dodge the name classifier."""
    if not value:
        return False
    return _URL_USERINFO_RE.search(value) is not None or _SECRET_VALUE_RE.search(value) is not None


def _build_bypass_env(workdir: str) -> dict[str, str]:
    """Env for bypass exec: full host env minus credential vars, with HOME at the workdir and TMPDIR
    just inside it so SDKs cannot read cached creds.

    Stripping the child env is necessary but not sufficient (a same-UID child can read the parent's
    env via procfs), so callers also harden the parent (see _harden_parent_against_proc_env_leak).
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
    # Repoint TEMP/TMP/TMPDIR so writes land in the session sandbox on every OS.
    env["TEMP"] = temp_dir
    env["TMP"] = temp_dir
    # Bypass inherits the operator's PYTHONPATH, so prepend.
    inherited_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (_SANDBOX_SITE_DIR, inherited_pythonpath) if part
    )
    # Windows SDKs read creds under profile dirs, not $HOME.
    for var in _BYPASS_ENV_WINDOWS_PROFILE_VARS:
        if var in os.environ:
            env[var] = workdir
    env.setdefault("MPLBACKEND", "Agg")
    return env


# Read by _sandbox_preexec after the fork, so it is resolved here first: the settings store cannot be read in the child.
_sandbox_as_bytes = 8 * 1024 * 1024 * 1024


def _refresh_sandbox_memory_limit() -> None:
    global _sandbox_as_bytes
    _sandbox_as_bytes = sandbox_memory_limit.memory_limit_bytes()


def _sandbox_preexec():
    """Best-effort sandbox setup for sandboxed subprocesses (modules are resolved at import time so
    the forked child runs no imports)."""
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

        # CLONE_NEWNET not applied: with userns it blocks allowlisted egress too. Network policy is
        # the AST host check and the bash blocklist.

    if _resource is not None:
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
            as_bytes = _sandbox_as_bytes
            if as_bytes is not None:
                _resource.setrlimit(_resource.RLIMIT_AS, (as_bytes, as_bytes))
        except (ValueError, OSError, AttributeError):
            pass
        try:
            cpu_s = int(os.environ.get("UNSLOTH_STUDIO_SANDBOX_CPU_S", "600"))
            _resource.setrlimit(_resource.RLIMIT_CPU, (cpu_s, cpu_s))
        except (ValueError, OSError, AttributeError):
            pass
        try:
            # Clamp to the inherited hard limit so setrlimit does not raise.
            nofile = int(os.environ.get("UNSLOTH_STUDIO_SANDBOX_NOFILE", "16384"))
            _soft_cur, hard_cur = _resource.getrlimit(_resource.RLIMIT_NOFILE)
            target = nofile if hard_cur == _resource.RLIM_INFINITY else min(nofile, hard_cur)
            _resource.setrlimit(_resource.RLIMIT_NOFILE, (target, target))
        except (ValueError, OSError, AttributeError):
            pass


def _account_confinement():
    """Confinement for the acting account's next child: None for owner, raises if unavailable."""
    return account_confinement(_SANDBOX_SITE_DIR)


def _run_preexecs(first, second):
    """Both pre-exec steps in the forked child, in order; no closures, no imports."""
    if first is not None:
        first()
    second()


def _apply_confinement(confinement, popen_kwargs: dict, argv: list) -> list:
    if confinement is None:
        return argv
    if confinement.preexec is not None and sys.platform != "win32":
        popen_kwargs["preexec_fn"] = partial(
            _run_preexecs, popen_kwargs.get("preexec_fn"), confinement.preexec
        )
    return confinement.wrap(argv)


def _bypass_preexec():
    """Minimal pre-exec for bypass exec: os.setsid() only. Required, not a restriction:
    _kill_process_tree does killpg(getpgid(child)), so without a new session a timeout/cancel
    would kill the Unsloth server too."""
    try:
        os.setsid()
    except OSError:
        pass


_last_tool_execution_record: "os_sandbox.ToolExecutionRecord | None" = None


def _note_tool_execution(record) -> None:
    """Built from a live probe, never from model output, so it is safe to show as a badge."""
    global _last_tool_execution_record
    if record is None:
        return
    _last_tool_execution_record = record
    logger.info("tool execution mode: %s", record.as_dict())


def _requested_execution_mode(tool_execution_mode: str, disable_sandbox: bool) -> str:
    """Require disable_sandbox for full access; checked here because managed accounts skip the planner."""
    if tool_execution_mode not in os_sandbox.TOOL_EXECUTION_MODES:
        raise os_sandbox.SandboxUnavailableError(
            f"TOOL_EXECUTION_MODE_INVALID: {tool_execution_mode!r} is not a tool "
            f"execution mode ({', '.join(os_sandbox.TOOL_EXECUTION_MODES)})",
            remediation = "Use 'auto' or 'required'.",
        )
    if disable_sandbox:
        return "full"
    if tool_execution_mode == "full":
        raise os_sandbox.SandboxUnavailableError(
            "TOOL_EXECUTION_MODE_INVALID: full access is not requestable through "
            "tool_execution_mode",
            remediation = "Full access is granted with disable_sandbox (Bypass Permissions).",
        )
    return tool_execution_mode


_with_session_packages = os_sandbox.with_session_packages


def _software_safeguards_launch(plan, fault: str):
    """Prepare an unisolated launch, retaining session packages and recording *fault*."""
    full = plan.requested_mode == "full"
    return os_sandbox.PreparedSandboxLaunch(
        argv = plan.argv,
        workdir = plan.workdir,
        env = _with_session_packages(plan.env, plan.workdir),
        preexec_fn = plan.preexec_fn,
        backend = "software-safeguards",
        timeout_seconds = plan.timeout_seconds,
        close_fds = plan.close_fds,
        terminate_descendants = plan.terminate_descendants,
        execution_record = os_sandbox.ToolExecutionRecord(
            requested_mode = plan.requested_mode,
            effective_mode = "full" if full else "software_safeguards",
            environment = sys.platform,
            backend = "software-safeguards",
            profile_id = "full-access" if full else "software-safeguards-v1",
            probe_generation = "",
            os_isolation = False,
            retained_safeguards = tuple(
                item
                for item in (
                    os_sandbox._FULL_SAFEGUARDS if full else os_sandbox._SOFTWARE_SAFEGUARDS
                )
                if item != "timeout" or plan.timeout_seconds is not None
            ),
            limitations = (
                ("security_restrictions_disabled", fault)
                if full
                else (*os_sandbox._software_only_limitations(), fault)
            ),
        ),
    )


def _reaches_host_paths(kind: str, text: str) -> bool:
    """Whether the call names a host path outside the silent roots: the check that put it in front of the user."""
    try:
        if kind == "python":
            tree, error = _parse_python(text)
            return error is None and _python_reaches_outside_sandbox(tree, text)
        decoded = _decode_ansi_c(text, keep_one_word = True)
        command = decoded.replace("\r\n", ";").replace("\n", ";").replace("\r", ";")
        for variant in {command, _expand_shell_assignments(_expand_param_defaults(command))}:
            lexer = shlex.shlex(variant, posix = True, punctuation_chars = ";&|()")
            lexer.whitespace_split = True
            if _terminal_reaches_outside_sandbox(list(lexer), variant):
                return True
    except Exception:  # noqa: BLE001 - unclassifiable stays isolated
        return False
    return False


def _prepare_tool_launch(plan, *, host_access_approved: bool = False):
    """Fall back only when the backend is unavailable; unsafe workdirs, build failures and (outside `full`) planner errors refuse."""
    if host_access_approved and plan.requested_mode == "auto":
        return _software_safeguards_launch(plan, "user_approved_host_access")
    try:
        prepared = os_sandbox.prepare_tool_launch(plan)
        if plan.preexec_fn is not None and prepared.preexec_fn is None:
            # Preserve setsid so timeout cleanup cannot kill the server's process group.
            logger.warning(
                "Sandbox backend %s dropped the launch pre-exec; restoring it",
                prepared.backend,
            )
            prepared.preexec_fn = plan.preexec_fn
        if prepared.execution_record is not None and not prepared.execution_record.os_isolation:
            prepared.env = _with_session_packages(prepared.env, plan.workdir)
        return prepared
    except (os_sandbox.WorkdirUnsafeError, os_sandbox.SandboxBuildError):
        # These failures can be tool-induced; fallback would let code remove its own boundary.
        raise
    except os_sandbox.SandboxUnavailableError:
        if plan.requested_mode == "required" or (
            plan.requested_mode not in os_sandbox.TOOL_EXECUTION_MODES
        ):
            raise
        logger.warning(
            "The sandbox backend is no longer available, running with software safeguards",
            exc_info = True,
        )
        return _software_safeguards_launch(plan, "sandbox_became_unavailable")
    except Exception as exc:  # noqa: BLE001 - construction failures never buy a host replay
        if plan.requested_mode == "full":
            return _software_safeguards_launch(plan, "sandbox_planner_error")
        raise os_sandbox.SandboxBuildError(
            f"sandbox preparation failed without host fallback: {exc}"
        ) from exc


_LAUNCHER_FAILURE_MARKERS = {
    "bubblewrap": "bwrap: ",
    "macos-seatbelt": "sandbox-exec: ",
}


def _forget_sandbox_capability_if_the_backend_failed(prepared, output: str) -> None:
    """Drop a stale probe verdict found at exec, so it costs one call rather than the cache's lifetime."""
    if prepared is None or prepared.backend == "software-safeguards":
        return
    if not output.startswith("Exit code "):
        return
    marker = _LAUNCHER_FAILURE_MARKERS.get(prepared.backend)
    if marker is None or marker not in output[:400]:
        return
    logger.warning("The sandbox backend failed at launch; re-probing the capability")
    try:
        from .sandbox_probe import reset_probe_cache
        reset_probe_cache()
        if sys.platform == "linux":
            from .sandbox_linux import reset_cache_verdicts
            reset_cache_verdicts()
    except Exception:  # noqa: BLE001 - a cache reset never breaks a tool result
        logger.debug("could not reset the sandbox probe cache", exc_info = True)


def _sandbox_refusal(exc) -> str:
    """The remediation is part of the answer: the reader can fix the host."""
    remediation = getattr(exc, "remediation", "") or ""
    return _truncate(f"Execution error: {exc}{(' ' + remediation) if remediation else ''}")


def _apply_prepared_launch(prepared, popen_kwargs: dict) -> dict:
    """The OUTER process needs its own session: every kill path is killpg based."""
    popen_kwargs["cwd"] = prepared.workdir
    popen_kwargs["env"] = prepared.env
    if sys.platform != "win32":
        popen_kwargs["preexec_fn"] = prepared.preexec_fn
    popen_kwargs["close_fds"] = prepared.close_fds
    if prepared.pass_fds:
        # Empty for every fallback, so Windows never sees a kwarg it rejects.
        popen_kwargs["pass_fds"] = tuple(prepared.pass_fds)
    return popen_kwargs


# PR_SET_DUMPABLE is process-global and sticky, so harden once.
_parent_proc_hardened = False


def _harden_parent_against_proc_env_leak() -> bool:
    """Make the Unsloth process's /proc/<pid>/environ unreadable to its children.

    Stripping the child env is not enough on Linux: a bypassed same-UID child can read
    /proc/<getppid()>/environ to recover the parent's unfiltered secrets. Clearing PR_SET_DUMPABLE
    reparents this process's /proc entries to root, closing that read.

    Returns True when hardened or unnecessary (off Linux), False when needed but unappliable (e.g.
    prctl denied by seccomp); callers must then fail closed. This is a mitigation, not a full
    boundary: a bypassed tool can still walk /proc to an ancestor or read creds by path. Applied
    lazily on first bypass.
    """
    global _parent_proc_hardened
    if _parent_proc_hardened:
        return True
    if sys.platform != "linux":
        return True
    if _libc is None:
        return False
    try:
        # ctypes returns -1 on failure and does not raise, so check it.
        ret = _libc.prctl(4, 0, 0, 0, 0)
    except (OSError, AttributeError):
        return False
    if ret != 0:
        return False
    _parent_proc_hardened = True
    return True


# These are the WSL launcher, which runs in the WSL filesystem; only native Win32 bash works.
_WSL_BASH_MARKERS = ("\\system32\\", "\\windowsapps\\")
# os.path.join so the resolver stays testable on POSIX.
_WIN_BASH_RELATIVE = (
    os.path.join("Git", "bin", "bash.exe"),
    os.path.join("Git", "usr", "bin", "bash.exe"),
)


def _is_trusted_windows_bash(path: str) -> bool:
    """True when ``path`` is a native Win32 bash under a system install root.

    The shell runs every sandboxed command, so an untrusted one defeats the PATH / PATHEXT /
    NoDefaultCurrentDirectoryInExePath hardening in _build_safe_env: a bash.exe in a user-writable
    dir (Scoop, a per-user Git, a project checkout) would execute attacker-controlled code for a
    command that already passed the blocklist. Same Program Files trust boundary as the sandbox git
    PATH entry (#7317), and fails closed.
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
    # shutil.which returns only the first match, which may be an untrusted shim.
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


def _windows_bash_git_cmd_dir() -> str:
    """The trusted ``Git\\cmd`` dir of the install the resolved bash belongs to, or ""."""
    bash = _windows_bash()
    if not bash:
        return ""
    bin_dir = os.path.dirname(bash)
    for root in (os.path.dirname(bin_dir), os.path.dirname(os.path.dirname(bin_dir))):
        candidate = os.path.join(root, "cmd")
        if (
            root
            and os.path.isfile(os.path.join(candidate, "git.exe"))
            and _is_trusted_windows_program_dir(candidate)
        ):
            return os.path.realpath(candidate)
    return ""


def _windows_bash_userland_dirs() -> list[str]:
    """Trusted dirs holding the resolved bash and the POSIX tools beside it.

    ``bash -c`` is non-login, so /etc/profile never runs and Git for Windows' ``usr\\bin`` stays off
    PATH, leaving ls / cat / grep command not found. bash.exe ships under ``Git\\bin`` or
    ``Git\\usr\\bin``, so both parents are probed. Candidates clear the same Program Files trust
    boundary as the git entry (#7317) and are canonicalised against junctions. Fails closed: no
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
        # cmd /c runs only the first line and mangles bash quoting; prefer a real bash.
        bash = _windows_bash()
        if bash:
            return [bash, "-c", command]
        return ["cmd", "/c", command]
    return ["bash", "-c", command]


def _windows_system_cmd() -> str:
    """System32 cmd.exe, resolved the same way MXC policy resolves it; never COMSPEC or PATH."""
    from .mxc_policy import _system_cmd
    return _system_cmd()


def _terminal_profile(disable_sandbox: bool = False) -> str:
    """Which shell the Terminal runs: "bash", "cmd_isolated" or "cmd_fallback".

    Git Bash cannot start inside MXC (microsoft/mxc#1061), so when bash fails the MXC probe and
    cmd.exe qualifies instead, Auto runs the Terminal isolated on cmd rather than unsandboxed on bash.
    Any bash failure counts, not only the MSYS verdict: on a freshly prepared host bash fails without
    that signature while cmd passes. Only a failed bash probes cmd. Full access and
    UNSLOTH_MXC_TERMINAL_CMD=0 keep the host shell.
    """
    if sys.platform != "win32":
        return "bash"
    bash = _windows_bash()
    host_default = "bash" if bash else "cmd_fallback"
    if disable_sandbox or os.environ.get("UNSLOTH_MXC_TERMINAL_CMD") == "0":
        return host_default
    try:
        if bash:
            verdict = os_sandbox.capability_snapshot(
                execution_kind = "terminal", selected_executable = bash
            )
            if verdict.available:
                return "bash"
        cmd = os_sandbox.capability_snapshot(
            execution_kind = "terminal", selected_executable = _windows_system_cmd()
        )
        return "cmd_isolated" if cmd.available else host_default
    except Exception as exc:  # noqa: BLE001 - a probe failure must never take the Terminal away
        logger.warning(f"terminal profile check failed, keeping the host shell: {exc}")
        return host_default


_request_profile: list = [None, 0.0]
_request_profile_lock = threading.Lock()
_REQUEST_PROFILE_REFRESH_SECONDS = 240.0
# Bumped by every reset: a refresh that started earlier must not publish the profile it computed.
_request_profile_generation = 0


def reset_terminal_profile_cache() -> None:
    """Forget the advertised Terminal profile, so the next request re-checks it (isolation settings changed)."""
    global _request_profile_generation
    with _request_profile_lock:
        _request_profile[:] = [None, 0.0]
        _request_profile_generation += 1


def _refresh_request_profile() -> str:
    with _request_profile_lock:
        generation = _request_profile_generation
    profile = _terminal_profile(False)
    with _request_profile_lock:
        if generation == _request_profile_generation:
            _request_profile[:] = [profile, time.monotonic()]
    return profile


def _profile_for_request() -> str:
    with _request_profile_lock:
        profile, computed = _request_profile
        stale = (
            profile is not None and time.monotonic() - computed > _REQUEST_PROFILE_REFRESH_SECONDS
        )
        if stale:
            _request_profile[1] = time.monotonic()
    if profile is None:
        return _refresh_request_profile()
    if stale:
        threading.Thread(target = _refresh_request_profile, daemon = True).start()
    return profile


def apply_terminal_profile_for_request(
    tools: list[dict], sandbox_level: "str | None" = None
) -> list[dict]:
    """Sandboxed requests only: advertise the shell _bash_exec will pick for this request. Only the
    first call can block on the MXC probe, so async callers run it in a worker thread; later calls
    reuse the last profile and refresh it in the background. A list without the Terminal never probes."""
    if not any(
        isinstance(t, dict) and (t.get("function") or {}).get("name") == "terminal"
        for t in tools or ()
    ):
        return tools
    profile = _terminal_profile(True) if sandbox_level == "low" else _profile_for_request()
    return apply_terminal_profile_description(tools, profile)


def _shell_argv(command: str, workdir: str, confinement) -> "tuple[list[str], str | None]":
    """Keep a confined account's command text off argv: other accounts' tools can read
    /proc/<pid>/cmdline and Landlock cannot deny per-pid reads."""
    argv = _get_shell_cmd(command)
    if confinement is None or sys.platform == "win32" or argv[1] != "-c":
        return argv, None
    fd, path = tempfile.mkstemp(suffix = ".sh", prefix = ".studio_cmd_", dir = workdir)
    with os.fdopen(fd, "w", encoding = "utf-8") as f:
        f.write(command)
    name = os.path.basename(path)
    return [argv[0], name], name


# Per-session workdirs; callers without a session_id get ~/studio_sandbox/_default.
_workdirs: dict[tuple[str, str], str] = {}
# A process whose cwd was removed fails every relative write.
_active_sessions: "dict[tuple[str, str], int]" = {}
# Deletions that arrived mid-call, keyed like the above, with every exact id folded onto it.
_pending_removals: "dict[tuple[str, str], dict[str, bool]]" = {}
_active_sessions_lock = threading.Lock()
# Starts for these wait on the condition, so only that chat is held up.
_removing_sessions: "set[tuple[str, str]]" = set()
_sessions_free = threading.Condition(_active_sessions_lock)


def _workdir_key(session_id: "str | None") -> tuple[str, str]:
    return current_account_id(), session_id or _ANON_KEY


def _session_key(session_id: "str | None") -> tuple[str, str]:
    """Lifecycle key for a session id.

    Case-folded: two ids differing only in case are one directory on Windows and
    on a default macOS volume, and keying them apart let a delete land while the
    other chat was running a tool in there.
    """
    return current_account_id(), (session_id or _ANON_KEY).casefold()


@contextlib.contextmanager
def _session_in_flight(session_id: "str | None"):
    key = _session_key(session_id)
    with _sessions_free:
        # A removal runs with the lock released; only this session waits.
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
        # Not an early return: that would swallow what the tool raised.
        if pending:
            # Outside the lock: the emptiness check can take seconds.
            try:
                for pending_id, pending_files in pending.items():
                    if _thread_exists(pending_id, unknown = True):
                        # Recreated meanwhile: the folder belongs to the new chat now.
                        continue
                    _remove_session_sandbox_locked(pending_id, pending_files)
            finally:
                with _sessions_free:
                    _removing_sessions.discard(key)
                    _sessions_free.notify_all()


_SESSION_ID_RE = re.compile(r"\A[A-Za-z0-9_\-]{1,64}\Z")
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
    """Where a preserved project workspace's path is written down. One small file per project, named
    by its id: the row that knew the path is gone, and a workspace the user pointed somewhere
    custom cannot be derived from anything else."""
    try:
        from utils.paths.storage_roots import account_path
        return str(account_path("orphaned-projects"))
    except Exception:
        # Only if the studio home is unresolvable.
        return os.path.join(
            os.path.dirname(os.path.realpath(sandbox_root())),
            "orphaned-projects",
        )


# Chats and projects can share a client id, so records are named by kind plus a digest.
_ORPHAN_CHAT = "chat"
_ORPHAN_PROJECT = "project"
# No cap: a cap would strand every record past it.
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
    """Remember where a deleted project's kept workspace lives. Written whether or not files were to
    be deleted: the row that knew the path has gone either way, and a chat forked out of the
    project still shows cards for it. ``pending_delete`` separates keep this, just make it
    reachable from the user asked for it, finish when nothing is using it."""
    if not project_id or not workspace:
        return
    _write_orphan_record(
        _ORPHAN_PROJECT,
        project_id,
        {
            "path": os.path.realpath(workspace),
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
    """Remember a chat sandbox kept because a fork still shows its files. The user asked for those
    files and the chat is gone, so nothing would come back to that folder: the fork's own delete
    finishes the job instead."""
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
    """Whether anything a record names is still on disk. The project root as well as its sandbox: a
    delete that removed the sandbox and stopped at a locked file elsewhere leaves the rest of the
    workspace, and dropping the record here loses both the path and the user's request."""
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

    Always through the storage helper, whose folder-name and denied-path checks decide what may go:
    a record is a file on disk, and a stale or edited one naming an unrelated directory must not
    become an rmtree of it. Without a recorded root the workspace's own parent is offered, which is
    what the default layout puts the sandbox in; anything else it refuses, and the record stays
    pending rather than being deleted on our own authority.
    """
    from storage.studio_db import delete_project_workspace

    target = root or os.path.dirname(os.path.realpath(workspace))
    delete_project_workspace({"id": project_id, "rootPath": target})


def collect_orphaned_project_workspaces() -> None:
    """Finish the workspace deletes the user asked for. Only records marked pending: one kept merely
    so a fork's cards resolve is not something anybody asked to remove. Skipped while a tool call
    is still running in there, or while a chat still shows its files."""
    from storage.studio_db import sandbox_is_referenced_elsewhere
    for record_id, workspace, root, pending, is_chat in list_orphaned_projects():
        if not pending:
            continue
        try:
            session = record_id if is_chat else project_session_id(record_id)
            # The id can be reused by a newer chat or project that now owns the folder.
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
                remove_session_sandbox(session, delete_files = True)
            else:
                _delete_recorded_workspace(record_id, workspace, root)
            # Transient failures keep the record so the next launch retries.
            forget_orphaned_project_if_gone(record_id, workspace, root, is_chat)
        except Exception:  # noqa: BLE001 - a stuck record must not break a delete
            logger.warning("Could not collect workspace for %s", record_id, exc_info = True)


def finish_workspace_delete_when_idle(
    project_id: str, timeout: float = 600.0
) -> "threading.Thread":
    """Wait out the tool call still using a workspace, then delete it. The delete dialog promised
    those files would go, and nothing else would come back to them: the collection otherwise runs
    only on the next delete."""

    def _wait_and_collect() -> None:
        session = project_session_id(project_id)
        wait_for_sessions_idle([session], timeout = timeout)
        collect_orphaned_project_workspaces()

    thread = account_thread(
        target = _wait_and_collect,
        name = "workspace-delete",
        daemon = True,
    )
    thread.start()
    return thread


def _recorded_project_workdir(project_id: str) -> "str | None":
    """The kept workspace of a deleted project, wherever the user put it. By key: a resolve happens
    on every tool call for such a project, and no number of other records may keep it from
    finding its own."""
    record = _read_orphan_record(_ORPHAN_PROJECT, project_id)
    if not record:
        return None
    path = record["path"]
    return path if os.path.isdir(path) else None


def _orphaned_project_workdir(project_id: str) -> "str | None":
    """A deleted project's workspace, when its files were kept. The record answers for any id, since
    it is keyed by a digest. Only the guess below builds a directory name, so only that needs an
    id a filename can hold."""
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
    """Whether a chat of the user's is stored under this exact id. ``unknown`` is what a check that
    could not be made returns: a caller about to delete files passes True, so a database hiccup
    keeps them, while one merely routing a call passes False and treats the id as a project's."""
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
    """Whether a project with this id is the one those folders belong to. A reused id is not the
    same workspace: the default root carries the project's name, and renaming a project leaves
    its root where it was. A folder the live row does not own is still the deleted project's, and
    still the one the user asked to remove."""
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
    """The project workspace a session id names, if it names one. The prefixed id can be longer than
    a directory name may be, or carry a character one may not: it is the project part that has to
    be usable, and the workspace path comes from the row rather than from the id."""
    if not session_id:
        return None
    if not _usable_session_id(session_id) and not session_id.startswith(_PROJECT_SESSION_PREFIX):
        return None
    return _get_project_workdir(session_id)


def _get_project_workdir(session_id: str) -> str | None:
    if not is_owner_context():
        return None
    if not session_id.startswith(_PROJECT_SESSION_PREFIX):
        return None
    project_id = session_id[len(_PROJECT_SESSION_PREFIX) :]
    if not project_id:
        return None
    if _thread_exists(session_id):
        # A chat may use this id too; sharing the project's workspace would run its tools there.
        return None
    try:
        from storage.studio_db import ensure_chat_project_workspace
        project = ensure_chat_project_workspace(project_id)
    except Exception:
        logger.warning("Failed to resolve project sandbox for %s", session_id, exc_info = True)
        return None
    if not project:
        # The project is gone but a forked chat still shows cards for this sandbox.
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


# The only evidence a directory is ours to delete; the root may be a shared folder.
_SANDBOX_MARKER = ".unsloth_sandbox"

# A derived name belongs to the id that hashes to it, never to a same-named chat.
_DERIVED_PREFIX = "_id-"

# A chat whose id is one of these gets a derived name instead of sharing that folder.
_FALLBACK_NAMES = frozenset({"_default", "_invalid"})

# Contains a char the id charset forbids, so no chat can key onto it.
_ANON_KEY = "\x00_default"


def _sandbox_name(session_id: str) -> str:
    """The directory name for an id. An id the filesystem cannot hold gets a name derived from it
    rather than a shared bucket: those ids come from API clients, and one bucket meant every such
    chat could read and delete every other one's files."""
    if (
        _usable_session_id(session_id)
        and not session_id.startswith(_DERIVED_PREFIX)
        and session_id not in _FALLBACK_NAMES
    ):
        return session_id
    # Ids that already look derived are derived too. surrogatepass: lone surrogates arrive from
    # JSON and surrogateescape.
    encoded = session_id.encode("utf-8", "surrogatepass")
    return _DERIVED_PREFIX + hashlib.sha256(encoded).hexdigest()[:16]


def _preserve_foreign_marker(workdir: str, name: "str | None" = None) -> None:
    """Move aside a marker-named entry that this migration did not write. This name was not reserved
    before the change, so a chat that wrote its own .unsloth_sandbox has a real file there, and a
    short note like notes reads as a perfectly good session name. Only the exact marker this move
    is about to write is left alone; everything else is renamed, not removed."""
    marker = os.path.join(workdir, _SANDBOX_MARKER)
    if not os.path.lexists(marker):
        return
    if name is not None and _marker_owner(workdir) == _sandbox_name(name):
        return
    for n in range(1, 100):
        kept = f"{marker}.saved" if n == 1 else f"{marker}.saved-{n}"
        if not os.path.lexists(kept):
            try:
                os.rename(marker, kept)
            except OSError:
                logger.warning("Could not preserve %s", marker)
            return


def _mark_sandbox(workdir: str, session_id: str) -> None:
    """(Re)write the marker. Never through a link: the file sits where tool code runs, so one
    replaced by a symlink would send this write to whatever it points at and truncate it."""
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
    """The id this directory was created for, when it says. Anything that is not an id reads as no
    owner rather than as somebody else's: a tool writing over this file would otherwise send its
    own chat to a fresh directory on the next launch, leaving its files behind."""
    marker = os.path.join(workdir, _SANDBOX_MARKER)
    if os.path.islink(marker):
        return None
    try:
        with open(marker, encoding = "utf-8") as fh:
            owner = fh.read(256).strip()
    except (OSError, UnicodeDecodeError):
        return None
    return owner if owner and _usable_session_id(owner) else None


def _session_dir(root: str, session_id: str) -> str:
    """The directory for this exact id.

    Two ids differing only in case are one name on Windows and on a default macOS volume. The marker
    says which id made the directory, and anyone else gets one of their own rather than sharing
    files that either chat's deletion would then remove. A directory already sitting in a root the
    user pointed us at is nobody's sandbox: it is stepped around rather than run in, so no tool can
    write a marker into it and make it look like ours.
    """
    name = _sandbox_name(session_id)
    plain = os.path.join(root, name)
    # A link is never ours; claiming through one writes into somebody else's directory.
    if not os.path.islink(plain):
        owner = _marker_owner(plain)
        if owner == name:
            return plain
        if owner is None and not (os.path.isdir(plain) and not _root_is_ours()):
            return plain
    return os.path.join(root, f"{name}-{_name_suffix(session_id)}")


def _name_suffix(session_id: str) -> str:
    """A short stable tail, so the same chat lands in the same directory. surrogatepass for the same
    reason _sandbox_name uses it: an id with a lone surrogate reaches here on the collision path,
    and a strict encode would raise rather than step aside."""
    encoded = session_id.encode("utf-8", "surrogatepass")
    return hashlib.sha256(encoded).hexdigest()[:8]


# Case-variant ids racing on a case-insensitive volume could both take one name.
_assign_lock = threading.Lock()


# Lets a marker a tool removed be written again; not a record of ownership.
_claimed_here: "set[str]" = set()


def _claim_sandbox(workdir: str, session_id: str) -> bool:
    """Write the marker if nobody has, and report whether this id owns it. O_EXCL, so of two
    processes creating the same directory exactly one claims it and the other is told to go
    elsewhere."""
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
        # The fallback name in a user-chosen root can be the user's own directory.
        if not os.path.exists(workdir):
            # A tree an interrupted migration left marked is this chat's.
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
        if _root_is_ours() and _marker_owner(workdir) is None:
            _mark_sandbox(workdir, session_id)
            return workdir
        # Somebody else's: take our own name; the fallback gets the same test.
        workdir = _free_fallback_dir(root, session_id)
        if workdir is None or not _contained_in_root(workdir, root):
            return _sandbox_fallback(root, "_invalid")
        os.makedirs(workdir, exist_ok = True)
        _claim_sandbox(workdir, session_id)
        return workdir


def _marked_sandbox_in(root: str, session_id: str) -> "str | None":
    """A directory in *root* whose marker names this chat, if there is one. A fresh fallback has a
    name nothing can recompute, so this is what finds it again on a later launch, on a read and
    on a delete. Bounded: a root the user pointed us at can hold a lot of their own folders."""
    name = _sandbox_name(session_id)
    # Only names derived from this id: tools can write any owner into the marker.
    candidates = [os.path.join(root, name), *_fallback_candidates(root, session_id)]
    # Listed once rather than a scan per candidate (33 names).
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
    """A name in *root* this chat may take, or None when they are all spoken for. The deterministic
    one first, so the same chat comes back to the same folder, then a fresh one rather than
    running inside anything already there."""
    # One we already made wins, so files do not scatter across launches.
    ours = _marked_sandbox_in(root, session_id)
    if ours:
        return ours
    for candidate in _fallback_candidates(root, session_id):
        if _free_for(candidate, _sandbox_name(session_id)):
            return candidate
    return None


# Recomputable names so a later launch finds the folder without a scan.
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
    """True unless the root is a directory the user pointed us at. A link is theirs as well:
    `<studio home>/sandbox` pointing somewhere else means the directories under it are the
    user's, and ours by construction is what lets a delete rename and remove one of them."""
    if (os.environ.get("UNSLOTH_STUDIO_SANDBOX_HOME") or "").strip():
        return False
    try:
        return not os.path.islink(sandbox_root())
    except OSError:
        return False


def _sandbox_is_ours(target: str) -> bool:
    """Ours by construction at our own root, otherwise only with the marker. A real file: isfile()
    follows a link, so a marker symlinked at any existing file would make an unrelated directory
    in a shared root deletable."""
    if _root_is_ours():
        return True
    marker = os.path.join(target, _SANDBOX_MARKER)
    return os.path.isfile(marker) and not os.path.islink(marker)


def _legacy_sandbox_root() -> str:
    """Where the sandbox used to live: a third folder in the user's home."""
    return os.path.join(os.path.expanduser("~"), "studio_sandbox")


def shared_sandbox_root() -> str:
    """The base every account's ``sandbox_root`` lives under; confinement hides it first."""
    override = (os.environ.get("UNSLOTH_STUDIO_SANDBOX_HOME") or "").strip()
    if override:
        return os.path.expanduser(override)
    try:
        from utils.paths.storage_roots import studio_root
        return str(studio_root())
    except Exception:
        return _legacy_sandbox_root()


def sandbox_root() -> str:
    """Root of the per-session tool sandboxes. Under the studio home, so UNSLOTH_STUDIO_HOME keeps
    everything in one place instead of leaving a stray ~/studio_sandbox. Falls back to the legacy
    path only if the studio root cannot be resolved."""
    override = (os.environ.get("UNSLOTH_STUDIO_SANDBOX_HOME") or "").strip()
    if override:
        if not is_owner_context():
            from utils.paths.storage_roots import external_account_sandbox_root
            return str(external_account_sandbox_root())
        return os.path.expanduser(override)
    try:
        from utils.paths.storage_roots import account_path
        return str(account_path("sandbox"))
    except Exception:
        if not is_owner_context():
            raise
        return _legacy_sandbox_root()


_legacy_sandbox_migrated = False
_legacy_sandbox_lock = threading.Lock()


_STAGING_SUFFIX = ".arriving-"


def _free_move_target(root: str, name: str) -> "str | None":
    """A name in *root* nothing occupies, for a legacy move to land on. The resolver's own answer
    first, so an untouched root keeps the plain name. Both derived names can be the user's in a
    root they pointed us at, and returning nothing there stranded the files at the legacy root
    for good: the marker the move writes is what finds this one again."""
    candidate = _session_dir(root, name)
    if not os.path.exists(candidate):
        return candidate
    if _marker_owner(candidate) == _sandbox_name(name):
        # Already moved; a duplicate legacy copy is left alone.
        return None
    if _root_is_ours():
        return None
    # Same derived names as the request path so both resolve to the same folder.
    for candidate in _fallback_candidates(root, name):
        if not os.path.exists(candidate):
            return candidate
    return None


def _staged_move(source: str, target: str, name: str) -> None:
    """Move one session in, so an interruption cannot look like a collision. Across filesystems
    shutil.move copies as it goes, and a run killed part way leaves a partial destination with
    the original still in place; the next launch would read that as a session the new root
    already has and strand the files. Filled under a name nothing resolves to, then renamed,
    which on one filesystem is atomic."""
    global _legacy_sandbox_migrated
    staging = f"{target}{_STAGING_SUFFIX}{uuid.uuid4().hex[:8]}"
    # Announced while neither end of the move is visible; migration-done checks must consult it.
    with _legacy_locks_guard:
        _legacy_moves_in_flight.add(name)
    try:
        try:
            shutil.move(source, staging)
        except OSError:
            shutil.rmtree(staging, ignore_errors = True)
            raise
        # Marked before the rename: across filesystems the legacy copy is already gone.
        _preserve_foreign_marker(staging, name)
        _mark_sandbox(staging, name)
        try:
            os.rename(staging, target)
        except OSError:
            # This tree is the only copy; put it back and let the next pass retry.
            try:
                os.rename(staging, source)
            except OSError:
                logger.warning("Sandbox %s left at %s: could not be moved in", name, staging)
            else:
                # A concurrent pass may have declared the migration finished; reopen it.
                _legacy_sandbox_migrated = False
            raise
        _mark_sandbox(target, name)
    finally:
        global _legacy_moves_done
        with _legacy_locks_guard:
            _legacy_moves_in_flight.discard(name)
            _legacy_moves_done += 1


# Bookkeeping only, never held across a move.
_legacy_one_lock = threading.Lock()

# Per-session so a first call does not wait out another chat's multi-GB copy.
_legacy_session_locks: "dict[str, threading.Lock]" = {}
_legacy_locks_guard = threading.Lock()


def _legacy_lock_for(name: str) -> threading.Lock:
    """The lock covering this one session's move."""
    with _legacy_locks_guard:
        return _legacy_session_locks.setdefault(name, threading.Lock())


# Held across a whole move: the only way to tell "arrived" from "in staging".
_legacy_moves_in_flight: "set[str]" = set()
# Count of finished moves: catches moves that started and ended inside a whole-tree pass.
_legacy_moves_done = 0


def _legacy_lock_peek(name: str) -> "threading.Lock | None":
    """The lock covering this session's move, if one was ever started. Never creates the entry: a
    name with nothing at the legacy root must not leave one behind."""
    with _legacy_locks_guard:
        return _legacy_session_locks.get(name)


# Holds several chats' files, so it is never moved up as one chat.
_LEGACY_SHARED_BUCKET = "_invalid"


def _legacy_names(session_id: str) -> "list[str]":
    """Every name this session's folder can have at the legacy root, which is also the key its move
    is locked under."""
    names = [_sandbox_name(session_id)]
    if not _usable_session_id(session_id):
        # Read in place, never moved or deleted, and only for such an id.
        names.append(_LEGACY_SHARED_BUCKET)
    elif session_id not in names and session_id not in _FALLBACK_NAMES:
        # Only the derived-prefix case; fallback names are nobody's chat.
        names.append(session_id)
    return names


def _marked_sandbox_after_moves(root: str, session_id: str) -> "str | None | bool":
    """_marked_sandbox_in with every legacy move of this session held off, or False when no move of
    it ever started. Under whichever name it was moved from: a chat whose id starts with the
    derived prefix moves under the literal id, and its staging tree is marked with the derived one."""
    locks = [_legacy_lock_peek(name) for name in _legacy_names(session_id)]
    locks = [lock for lock in locks if lock is not None]
    if not locks:
        return False
    # Take all at once in a fixed order; a mover holds only one, so no deadlock.
    with contextlib.ExitStack() as held:
        for lock in locks:
            held.enter_context(lock)
        return _marked_sandbox_in(root, session_id)


def _legacy_session_dir(session_id: str) -> "str | None":
    """This session's directory at the legacy root, while one is still there.

    Both names, like the migration itself: a chat from before the upgrade whose
    id starts with the derived prefix kept its folder under the literal id.
    """
    if not is_owner_context():
        return None
    legacy_root = _legacy_sandbox_root()
    for name in _legacy_names(session_id):
        candidate = os.path.join(legacy_root, name)
        if os.path.islink(candidate):
            continue
        if os.path.isdir(candidate):
            lock = _legacy_lock_for(name)
        else:
            # Gone from legacy may mean in staging; wait on the lock to let the move land.
            lock = _legacy_lock_peek(name)
            if lock is None:
                continue
        # Re-checked under the move lock: a mid-move path lists nothing.
        with lock:
            if os.path.isdir(candidate) and not os.path.islink(candidate):
                return candidate
    return None


def _migrate_one_legacy_session(root: str, name: str) -> None:
    """Bring one session up from the legacy root, without waiting for the rest."""
    if not is_owner_context():
        return
    # Ask the legacy root, not the done flag or session dir: during a staged move neither root
    # holds the tree, and the root stays present for any move or failure.
    legacy_root = _legacy_sandbox_root()
    if not os.path.isdir(legacy_root):
        return
    source = os.path.join(legacy_root, name)
    if os.path.islink(source):
        return
    if os.path.isdir(source):
        lock = _legacy_lock_for(name)
    else:
        # An entry is the durable trace of a move (entries are never removed). Peeking without
        # inserting keeps the table bounded.
        lock = _legacy_lock_peek(name)
        if lock is None:
            return
    with lock:
        if not os.path.isdir(source):
            return
        # Through the resolver: at a shared root the plain name can be the user's own.
        target = _free_move_target(root, name)
        if target is None or not _contained_in_root(target, root):
            return
        try:
            os.makedirs(root, exist_ok = True)
            _staged_move(source, target, name)
        except OSError as error:
            logger.warning("Could not move sandbox %s: %s", name, error)


_legacy_background: "threading.Thread | None" = None


def _start_legacy_migration() -> "threading.Thread | None":
    """Carry the rest of the tree up, one pass at a time, off this request."""
    global _legacy_background
    if not is_owner_context() or _legacy_sandbox_migrated:
        return None
    with _legacy_one_lock:
        if _legacy_background is not None and _legacy_background.is_alive():
            return _legacy_background
        _legacy_background = migrate_legacy_sandbox_in_background()
        return _legacy_background


def _migrate_legacy_sandbox(root: str) -> None:
    """Move sessions from ~/studio_sandbox into the studio home, once. Those files are the user's,
    so they move rather than being dropped. A session already present at the new root wins and
    its legacy copy is left alone, so nothing is silently overwritten."""
    global _legacy_sandbox_migrated
    if not is_owner_context() or _legacy_sandbox_migrated:
        return
    # Flagged only once done, or a concurrent call creates the destination.
    with _legacy_sandbox_lock:
        if _legacy_sandbox_migrated:
            return
        # Only when nothing movable is left; committed with the check so a rollback is not overwritten.
        _migrate_legacy_sandbox_locked(root)


def _migrate_legacy_sandbox_locked(root: str) -> bool:
    """True when the legacy root holds nothing that could still be moved. A collision is not a
    failure: the new root already has that session, and the legacy copy is deliberately left for
    the user to find."""
    global _legacy_sandbox_migrated
    legacy = _legacy_sandbox_root()
    try:
        with _legacy_locks_guard:
            moves_before = _legacy_moves_done
        if os.path.realpath(legacy) == os.path.realpath(root) or not os.path.isdir(legacy):
            _legacy_sandbox_migrated = True
            return True
        os.makedirs(root, exist_ok = True)
        moved = 0
        own_moves = 0
        complete = True
        for name in os.listdir(legacy):
            source = os.path.join(legacy, name)
            # A link is not our sandbox; the marker would land outside both roots.
            if os.path.islink(source) or not os.path.isdir(source):
                continue
            if name == _LEGACY_SHARED_BUCKET:
                continue
            # Same resolver as the request path, so no chat's files are stranded.
            target = _free_move_target(root, name)
            if target is None or not _contained_in_root(target, root):
                continue
            try:
                with _legacy_lock_for(name):
                    if not os.path.isdir(source) or os.path.exists(target):
                        continue
                    own_moves += 1  # before the call: one that raises still advances the count
                    _staged_move(source, target, name)
                moved += 1
            except OSError as error:
                # Locked Windows files are retryable: report unfinished.
                complete = False
                logger.warning("Could not move sandbox %s: %s", name, error)
        if moved:
            logger.info("Moved %d chat sandbox folder(s) from %s to %s", moved, legacy, root)
        # Any other move overlapping this pass makes the listing stale: report unfinished and keep
        # the legacy root so a rollback has a place to land.
        with _legacy_locks_guard:
            overlapped = (
                bool(_legacy_moves_in_flight) or _legacy_moves_done - moves_before != own_moves
            )
            if not overlapped and complete:
                _legacy_sandbox_migrated = True
        if overlapped:
            return False
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
    """``_default`` / ``_invalid`` under the root, contained like any session. They are ordinary
    directories in a writable sandbox, so one replaced by a symlink would otherwise become the
    root every unchecked request reads from. Dropping that link is only ours to do at our own
    root; in a directory the user pointed us at, the entry is theirs and a fresh one is used
    instead."""
    owner = _sandbox_name(name)
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
    # In a user-chosen root an existing directory with this name is theirs.
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
        # The entry must be free and the claim must succeed (exist_ok would follow links).
        if not _free_for(made, owner):
            continue
        try:
            os.makedirs(made, exist_ok = True)
        except OSError:
            continue
        if _claim_sandbox(made, name):
            return made
    return _nothing_to_serve(name)


# Empty dir outside every root, so listings are empty and downloads 404.
_NOTHING_ROOT = None
_nothing_lock = threading.Lock()


def _nothing_to_serve(name: str) -> str:
    """A path that exists nowhere the user keeps files. The name is derived first: callers pass the
    id straight from the request, and an absolute one like ``/etc`` would make os.path.join drop
    the root it was given and hand back a directory of the system's."""
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
    return resolved if _contained_in_root(resolved, root) else root


def _contained_in_root(workdir: str, root: str) -> bool:
    """Whether a resolved session path is still inside the sandbox root. Applied to cached paths
    too: a directory replaced by a symlink after it was cached would otherwise keep serving from
    wherever it now points."""
    try:
        resolved, base = os.path.realpath(workdir), os.path.realpath(root)
        # commonpath, not prefix: a filesystem root already ends in a separator.
        return resolved != base and os.path.commonpath([resolved, base]) == base
    except (OSError, ValueError):
        return False


def _owned_by_session(workdir: str, session_id: str) -> bool:
    """Whether this session may read *workdir*, for a caller that creates nothing.
    ``_ensure_session_dir`` claims or steps aside; a read has to decide on what is already there,
    and the name it was given can be somebody else's too."""
    owner = _marker_owner(workdir)
    if owner is not None:
        return owner == _sandbox_name(session_id)
    # No marker, so the name is the only evidence; case-insensitive on Windows/macOS.
    return _root_is_ours() and os.path.basename(workdir) == _sandbox_name(session_id)


def _get_workdir(session_id: str | None = None) -> str:
    """Return a per-session sandbox dir at mode 0o700."""
    global _workdirs
    key = _workdir_key(session_id)
    cached = _workdirs.get(key)
    if cached is not None and not os.path.isdir(cached):
        cached = None
    if cached is not None and not _get_project_workdir(session_id or ""):
        # The entry may have been swapped for a link since; recheck like a fresh resolve.
        root_now = sandbox_root()
        # A tool can delete the marker; rewrite it for a directory this run claimed.
        if (
            session_id
            and cached in _claimed_here
            and not os.path.islink(cached)
            and _contained_in_root(cached, root_now)
            and os.path.isdir(cached)
            and _marker_owner(cached) != _sandbox_name(session_id)
        ):
            # This process made it for this chat, whatever the marker says now.
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
        # Before anything below creates a directory: a call after deletion must refuse.
        ensure_dir(Path(sandbox_root_path))
        # Migrate only this chat's legacy folder so a first call never waits on the whole tree.
        if session_id:
            # Pre-upgrade chats with a derived-prefix id kept their literal folder.
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
            _claim_sandbox(workdir, "_default")
        # Only a root we just created: the override can name a shared directory.
        if not root_existed or not (os.environ.get("UNSLOTH_STUDIO_SANDBOX_HOME") or "").strip():
            try:
                os.chmod(sandbox_root_path, 0o700)
            except OSError:
                pass
        # Only ours: a shared root may hold a same-named directory.
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
    """Where a session's sandbox would be, without creating it. For read-only callers: serving a
    file must not materialise a directory for every id someone asks about."""
    if session_id:
        project = _project_workdir_for(session_id)
        if project:
            return project
    root = sandbox_root()
    cached = _workdirs.get(_workdir_key(session_id))
    if (
        cached
        and not os.path.islink(cached)
        and _contained_in_root(cached, root)
        and (not session_id or _owned_by_session(cached, session_id))
    ):
        return cached
    if not session_id:
        return _sandbox_fallback(root, "_default")
    # This process made it for this chat, whatever a tool wrote into the marker since.
    claimed = _claimed_by_this_run(session_id, os.path.realpath(root))
    if claimed:
        return claimed
    workdir = _session_dir(root, session_id)
    if not _contained_in_root(workdir, root):
        return _sandbox_fallback(root, "_invalid")
    if not os.path.isdir(workdir):
        # A move that could not rename into place leaves the only copy under a marked name.
        ours = _marked_sandbox_in(root, session_id)
        if ours and _STAGING_SUFFIX in os.path.basename(ours):
            # May be a staging tree mid-move; once the session lock is free a remaining one is stranded.
            settled = _marked_sandbox_after_moves(root, session_id)
            if settled is not False:
                ours = settled
        if ours:
            return ours
        # The background move can take minutes; read from the legacy root meanwhile.
        legacy = _legacy_session_dir(session_id)
        if legacy:
            return legacy
        if not os.path.isdir(workdir):
            # A move that both failed to rename and to roll back leaves a staging tree the first scan missed.
            ours = _marked_sandbox_after_moves(root, session_id)
            if ours:
                return ours
    if not _root_is_ours() and not _owned_by_session(workdir, session_id):
        # In a user-chosen root the chat may be in a fallback name nothing recomputes.
        ours = _marked_sandbox_in(root, session_id)
        if ours:
            return ours
    if os.path.isdir(workdir) and not _owned_by_session(workdir, session_id):
        return _nothing_to_serve(session_id)
    return workdir


def migrate_legacy_sandbox_in_background() -> "threading.Thread":
    """Move the legacy sandbox up at startup, off every request. Across filesystems this copies
    every session, which is not something a listing or a download can wait on: those run on the
    event loop."""

    def _run() -> None:
        try:
            _migrate_legacy_sandbox(sandbox_root())
        except Exception:  # noqa: BLE001 - best effort, like the rest of this
            logger.debug("legacy sandbox migration failed", exc_info = True)

    thread = account_thread(target = _run, name = "sandbox-migrate", daemon = True)
    thread.start()
    return thread


_DETACHED_SUFFIX = ".deleting-"
# Exact shape so a user's `report.deleting-old` does not match.
_DETACHED_RE = re.compile(r"\A.+\.deleting-[0-9a-f]{8}\Z")


# One worker, not a thread per chat.
_delete_queue: "queue.Queue[tuple[str, int]]" = queue.Queue()
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
    """Queue another attempt at a tree ignore_errors left behind. A file held open by a scanner or a
    process still exiting is transient, and on Windows routine. The route has already told the
    user those files went, so waiting for the next launch's sweep is not an answer."""
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
                # No thread available: delete synchronously rather than leave an orphan tree.
                _delete_worker = None
                shutil.rmtree(target, ignore_errors = True)
                return
    _delete_queue.put((target, 0))


def sweep_detached_sandboxes(root: "str | None" = None) -> None:
    """Finish deletes a previous run was killed part way through.

    The rename is what puts the tree out of reach, so a kill between it and the rmtree leaves a full
    copy of the files nothing resolves to.

    At our own root the name is enough: nothing but this code puts a ``.deleting-<hex>`` directory
    there, and a tool that had removed the marker before the delete would otherwise leave the tree
    unreachable for good. In a root the user pointed us at, the marker is still required, since a
    folder of theirs can carry any name.
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
_swept_detached_accounts: set[str] = set()


def start_sandbox_recovery() -> "threading.Thread | None":
    """Finish what an interrupted run left: renamed trees and pending deletes."""
    return _start_detached_sweep()


def _start_detached_sweep() -> "threading.Thread | None":
    """Run the sweep once per process, off the call that noticed."""
    global _swept_detached
    with _legacy_one_lock:
        if is_owner_context():
            if _swept_detached:
                return None
            _swept_detached = True
        else:
            account_id = current_account_id()
            if account_id in _swept_detached_accounts:
                return None
            _swept_detached_accounts.add(account_id)

    def _sweep() -> None:
        sweep_detached_sandboxes()
        collect_orphaned_project_workspaces()

    thread = account_thread(target = _sweep, name = "sandbox-sweep", daemon = True)
    thread.start()
    return thread


def remove_session_sandbox(session_id: str, delete_files: bool = False) -> bool:
    """Drop a deleted chat's sandbox. True when something was removed.

    The chat was the only handle on that directory, so leaving it behind means one unreachable
    folder per chat forever. Empty folders always go; files need ``delete_files``, since they are
    the user's and are downloadable from the chat. Project workspaces are shared and have their own
    delete flow.
    """
    if not session_id:
        return False
    # Only sessions that really resolve to a project workspace.
    if session_id.startswith(_PROJECT_SESSION_PREFIX) and _get_project_workdir(session_id):
        # Unless this id has its own sandbox: the row goes first and the files would stay behind.
        root_here = os.path.realpath(sandbox_root())
        if not _claimed_by_this_run(session_id, root_here) and not _marked_sandbox_in(
            root_here,
            session_id,
        ):
            return False
    # Migrate only this session's legacy folder, outside the lock below.
    root_now = sandbox_root()
    _migrate_one_legacy_session(root_now, _sandbox_name(session_id))
    if _usable_session_id(session_id) and _sandbox_name(session_id) != session_id:
        _migrate_one_legacy_session(root_now, session_id)
    _start_legacy_migration()
    # Held across decision and unlink so no tool starts in between.
    key = _session_key(session_id)
    with _sessions_free:
        while key in _removing_sessions:
            _sessions_free.wait()
        if _active_sessions.get(key, 0) > 0:
            # Queued: the chat is gone from history, so nothing will name it again.
            queued = _pending_removals.setdefault(key, {})
            queued[session_id] = delete_files or queued.get(session_id, False)
            return False
        # Released while the tree is walked so unrelated chats are not blocked.
        _removing_sessions.add(key)
    try:
        return _remove_session_sandbox_locked(session_id, delete_files)
    finally:
        with _sessions_free:
            _removing_sessions.discard(key)
            _sessions_free.notify_all()


def session_sandbox_has_files(session_id: str) -> bool:
    """Whether this chat's sandbox still holds files of the user's. For a delete that was not
    offered the choice: the chat was the only way to those files, so the caller can offer it
    afterwards rather than leave a folder nothing can reach."""
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
    """Whether ``parent/name`` is a spill this process wrote, rather than anything else. Both halves
    are checked: the name has to be one `_spill_full_output` generates, and it has to sit at the
    spill root or one scope below it. A link is never one, whatever it is called."""
    root = os.path.join(sandbox, _SPILL_DIR)
    if parent != root and os.path.dirname(parent) != root:
        return False
    identity, owned = _spill_record(root)
    if identity is None or identity != _spill_identity(root):
        # No record: everything in it is the user's.
        return False
    return _is_recorded_spill(root, os.path.join(parent, name), owned)


def _holds_no_user_files(target: str, owner: "str | None" = None) -> bool:
    """Whether a sandbox holds nothing but (possibly empty) directories. Our own marker does not
    count, and only while it is still ours: tool code runs in there and can write its own content
    over that file, which is then the only copy of it. Bounded like every other walk here, and a
    tree too big to check is not one to remove without being asked."""
    budget = _MAX_SNAPSHOT_DIRS
    for parent, dirs, files in os.walk(target):
        if any(os.path.islink(os.path.join(parent, name)) for name in dirs):
            return False
        for name in files:
            if parent == target and name in _INTERNAL_SANDBOX_FILES:
                if name != _SANDBOX_MARKER:
                    continue
                marker = _marker_owner(target)
                if marker is not None and owner in (None, marker):
                    continue
            # Spill artifacts are Unsloth's own truncated output, not user content; matched by the exact
            # name `_spill_full_output` generates.
            if _is_spill_artifact(target, parent, name):
                continue
            if _is_attachment_copy(target, parent, name):
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
    cached = _workdirs.get(_workdir_key(session_id))
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
    """Wait until no tool call is running for these sessions. True if none is. Cancelling a
    generation only sets its event, so the call inside the executor is still using its working
    directory for a moment after."""
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
    """Whether this session's removal is queued behind a running tool call. The caller reports what
    it kept, and the answer is not known yet: the call still in flight can write a file after the
    sandbox looked empty, and the deferred removal would then keep it with nobody left to say so."""
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
        if not _owned_by_session(entry, session_id):
            entry = _marked_sandbox_in(root, session_id) or claimed or entry
    # The entry itself: a symlink to a sibling would take that chat's files. Drop links only at
    # our own root.
    if os.path.islink(entry):
        if not _root_is_ours():
            return False
        try:
            os.unlink(entry)
            return True
        except OSError:
            return False
    target = os.path.realpath(entry)
    if os.path.dirname(target) != root or not os.path.isdir(target):
        return False
    ours_here = bool(claimed) and target == os.path.realpath(claimed)
    if not ours_here and not _sandbox_is_ours(target):
        return False
    # Made for a different id: the same directory on a case-insensitive volume, and those files are the other chat's.
    owner = _marker_owner(target)
    if owner is not None and owner != _sandbox_name(session_id):
        return False
    if owner is None and not ours_here and os.path.basename(target) != _sandbox_name(session_id):
        # Without a marker the name is the only evidence, and it names the other chat.
        return False
    _workdirs.pop(_workdir_key(session_id), None)
    # Resolve the record path before removal; it cannot be derived once the tree is gone.
    forget_record = _spill_record_path(os.path.join(target, _SPILL_DIR))
    try:
        if delete_files:
            # Rename under the lock, delete after, so a large rmtree does not stall every chat.
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
        # Empty means no user files; empty dirs alone do not count.
        if not _holds_no_user_files(target, _sandbox_name(session_id)):
            return False
        shutil.rmtree(target, ignore_errors = True)
        gone = not os.path.isdir(target)
        if gone:
            _forget_spill_record(forget_record)
        return gone
    except OSError:
        return False


# Exact-string replacement, not a unified diff: models corrupt hunk headers. A missing or
# non-unique old_string is a hard error naming the match count.

_EDIT_FILE_MAX_BYTES = _env_int("UNSLOTH_STUDIO_EDIT_FILE_MAX_BYTES", 16 * 1024 * 1024)

# Characters are capped per line and overall: one line of minified JS can be the whole file.
_EDIT_FILE_DIFF_LINES = 40
_EDIT_FILE_DIFF_LINE_CHARS = 200
_EDIT_FILE_DIFF_CHARS = 4000
# Only a window goes to difflib; the whole file would be millions of line strs.
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
    from state.tool_policy import require_tool_access

    require_tool_access(disable_sandbox = disable_sandbox)
    raw = (raw_path or "").strip()
    if not raw:
        return None, "Error: 'path' is required."
    workdir = _get_workdir(session_id)
    candidate = raw
    # An absolute path already inside the workdir is real, not a habit to strip.
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
    # Full access runs tools unsandboxed already; confining this one only pushes models to cat.
    if not disable_sandbox and _is_outside_workdir(target, workdir):
        return None, (
            f"Error: '{raw}' is outside this conversation's working directory, "
            "which is the only place edit_file can write. Use a relative path "
            f"(for example '{os.path.basename(raw) or 'file.py'}')."
        )
    return target, ""


def _edit_file_decode(data: bytes, path: str) -> "tuple[str, str, str, str]":
    """Decode file bytes into (text, newline, bom, error). ``text`` is normalized to \\n so an
    old_string with plain newlines still matches a CRLF file; matching raw bytes would fail every
    edit of a Windows-authored source. The original convention is returned so the write puts it
    back instead of converting every line ending in the file."""
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
    # Judged against the total so a few stray CRs still write back as LF.
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

    Sibling temp file then rename: an interrupted write must not leave a source file half-replaced.
    The mode is carried over so an edit keeps the executable bit.

    ``expect`` are the bytes the edit was computed from, compared again here so a file another chat
    rewrote meanwhile is not reverted to this call's stale copy. ``workdir`` re-checks containment
    just before the rename, so a parent swapped for a symlink after the path was resolved is caught.
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
            pass
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
        from core import library

        library.replace_file(tmp, path)
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
    """A bounded unified diff of what changed. Line-numbered so the model can confirm the edit
    landed where it meant. Two separate bounds, because either alone leaks: difflib sees only a
    window around the first change, and the generator is consumed lazily."""
    import difflib
    import itertools

    window_start, window_end = _edit_file_line_window(
        before, change_at, _EDIT_FILE_DIFF_WINDOW_LINES
    )
    # Replay the replacement on the old window rather than cutting an equal-length window from
    # the new text, which misaligns when the edit changes line count.
    window_end = max(window_end, change_at + len(old))
    before_window = before[window_start:window_end]
    after_window = before_window.replace(old, new)
    first_line = before.count("\n", 0, window_start) + 1
    stream = difflib.unified_diff(
        before_window.split("\n"),
        after_window.split("\n"),
        lineterm = "",
        n = 2,
    )
    taken = list(itertools.islice(stream, 2 + _EDIT_FILE_DIFF_LINES + 1))[2:]
    plural = "" if count == 1 else "s"
    head = f"Edited {name} ({count} replacement{plural})"
    if not taken:
        return head
    if len(taken) > _EDIT_FILE_DIFF_LINES:
        diff = taken[:_EDIT_FILE_DIFF_LINES] + ["... (more diff lines)"]
    else:
        diff = taken
    diff = [
        line
        if len(line) <= _EDIT_FILE_DIFF_LINE_CHARS
        else f"{line[:_EDIT_FILE_DIFF_LINE_CHARS]}... (+{len(line) - _EDIT_FILE_DIFF_LINE_CHARS} chars)"
        for line in diff
    ]
    # Shift hunks from window to real file line numbers.
    if first_line > 1:
        diff = [_edit_file_shift_hunk(line, first_line - 1) for line in diff]
    body = "\n".join(diff)
    if len(body) > _EDIT_FILE_DIFF_CHARS:
        body = body[:_EDIT_FILE_DIFF_CHARS] + "\n... (receipt truncated)"
    return head + "\n" + body


def _edit_file_replace_all(value: object) -> "bool | None":
    """Read replace_all strictly; None means not a boolean. bool("false") is True, and models do
    emit the JSON string. Coercing it that way turns the multi-match guard off and rewrites every
    occurrence."""
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

    A zero-byte file is writable here on purpose: refusing every existing target would strand the
    model, since an empty old_string would be refused and no other old_string can match an empty
    file.

    The absent case is created with O_EXCL rather than checked and then written: two chats sharing a
    workspace can both pass a lexists() check and the later rename drops the earlier file. O_EXCL
    also gives the new file the usual umask-derived mode, where a mkstemp temp file would leave it
    0600.
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
            # A partial write after ENOSPC cannot be retried (empty old_string refuses non-empty), so
            # remove the inode.
            with contextlib.suppress(OSError):
                os.remove(target)
            return f"Error: cannot write '{name}': {exc}"
        return f"Created {name} ({new.count(chr(10)) + 1 if new else 0} lines)"
    try:
        st = os.stat(target)
    except OSError:
        st = None
    # Refused: a FIFO reports size 0 and opening it with no writer blocks forever.
    if st is None or not S_ISREG(st.st_mode) or st.st_size:
        return (
            f"Error: '{name}' already exists. An empty 'old_string' only "
            "creates a new file; to change this one, pass the exact text to "
            "replace."
        )
    error = _edit_file_write(target, new, newline, "", expect = b"", workdir = workdir)
    if error:
        return error
    return f"Created {name} ({new.count(chr(10)) + 1 if new else 0} lines)"


# Each entry is a full scan of up to 16 MiB; bounds a model-generated batch.
_MAX_EDITS_PER_CALL = 100

# Bounds the matches one replace_all may expand into.
_MAX_MATCH_SPANS = 10_000


def _edit_file_parse_edits(raw) -> "tuple[list[tuple[str, str, bool]], str]":
    """Validate the `edits` array into `(old, new, replace_all)` triples.

    Every string is normalized the way the file is, so a snippet copied out of `cat` output with
    plain newlines still matches a Windows-authored file.

    The surrogate check is per entry and up front, before anything is written: a truncated emoji
    escape survives `json.loads` as a lone surrogate that cannot be encoded, and the
    UnicodeEncodeError the write raises is swallowed upstream into Unknown tool: edit_file, the one
    answer that sends the model back to the whole-file rewrite. `old_string` needs no check, being
    only ever compared.
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
        # Checked, not coerced: str(None) would write the literal None into a file.
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

    Matched against `before` rather than against the running result, which is the rule llama.cpp's
    own `edit_file` states and the only one a model can reason about: it copied each `old_string`
    out of the file it read, so an entry that silently matched the output of an earlier entry would
    land somewhere it never saw.

    Spans are resolved for all entries first and checked for overlap, then applied right to left so
    the earlier offsets stay valid. Every failure returns before a single byte is written: a partly
    applied batch is the one outcome worse than a refused one, because the model cannot tell which
    half landed.
    """
    spans: list[tuple[int, int, str, int]] = []
    for index, (old, new, replace_all) in enumerate(edits, 1):
        if not old:
            # Defence in depth: `find` cannot advance on a zero-length pattern and would hang.
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
        # One entry needs no spans; str.replace avoids millions of span tuples.
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
    # One pass with a cursor; slice-and-concat per span is quadratic.
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
    # The receipt echoes file content, so an edit is a read, and Bypass lifts containment.
    if _references_studio_credential(str(arguments.get("path") or "")):
        return _STUDIO_CREDENTIAL_BLOCKED
    edits, error = _edit_file_parse_edits(arguments.get("edits"))
    if error:
        return error
    target, error = _edit_file_resolve(
        str(arguments.get("path") or ""), session_id, disable_sandbox
    )
    if error:
        return error
    # Again on the resolved path: `../../auth/...` only names it after the join.
    if _references_studio_credential(target) or _references_studio_credential(
        os.path.realpath(target)
    ):
        return _STUDIO_CREDENTIAL_BLOCKED
    name = os.path.basename(target)
    # Before the no-op check: both strings empty is how a zero-byte file is created.
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
    # FIFOs and character devices never end and this path has no timeout.
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


_WEB_SEARCH_QUERY_ALIASES = ("query", "q", "search_query", "search", "text")
_WEB_SEARCH_URL_ALIASES = ("url", "uri", "href", "link")


def _first_nonempty_arg(arguments: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _resolve_web_search_args(arguments) -> tuple[str, str]:
    args = arguments if isinstance(arguments, dict) else {}
    return (
        _first_nonempty_arg(args, _WEB_SEARCH_QUERY_ALIASES),
        _first_nonempty_arg(args, _WEB_SEARCH_URL_ALIASES),
    )


def canonicalize_web_search_arguments(arguments) -> dict:
    args = dict(arguments) if isinstance(arguments, dict) else {}
    query, url = _resolve_web_search_args(args)
    if url:
        return {"url": url}
    canonical: dict = {}
    if query:
        canonical["query"] = query
    if "image_queries" in args:
        canonical["image_queries"] = args["image_queries"]
    return canonical


def web_search_tool_with_images() -> dict:
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
    """Platform and working-directory note, on BOTH tool descriptions. Models habitually write to
    /mnt/data, a ChatGPT code-interpreter path that does not exist here, so the POSIX text names
    it. Naming only POSIX paths on Windows reads as you are on Linux and models then refuse to
    invoke Windows programs that are in fact available, so that text says where the code runs
    instead: without it a model assumes the pipe is its only output and declines to open a window
    it believes nobody can see."""
    # Otherwise a model assumes its output vanished into a scratch dir.
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


# Full access edits the sandboxed text rather than keeping a drifting second copy.
# test_full_access_tool_prompt.py checks the sandboxed markers are gone.
_FULL_ACCESS_SUBSTITUTIONS = (
    ("Execute Python code in a sandbox and", "Execute Python code and"),
    (
        "; absolute paths like /mnt/data or /tmp/outputs do not exist.",
        ". This runs wherever Unsloth Studio is running, which may be a remote host "
        "or a container with only some paths mounted.{clause}",
    ),
    # Windows never denies absolute paths, so state the capability. Tools run on the host serving
    # Unsloth, which may not be the user's device (--secure, -H 0.0.0.0).
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


# What sandbox-off means for paths. Both tools keep the sitecustomize shim on PYTHONPATH,
# so python rewrites missing convention-prefix paths (/mnt/data) under the workdir; the terminal
# follows the shell's rules except for Python it launches.
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

    Under bypass_permissions the loops pass disable_sandbox=True: _build_bypass_env /
    _bypass_preexec skip the static analysis, the command blocklist and the rlimits, so the host
    filesystem really is reachable. Handing a model the sandboxed text in that mode makes it answer
    I am sandboxed and cannot see your files to a question one tool call would have answered.
    Untouched clauses are the ones still true in both modes: the workdir is the per-session dir
    either way (_build_bypass_env repoints HOME at it and TMPDIR / TEMP / TMP just inside it), and
    so is the download-link note.
    """
    clause = _FULL_ACCESS_CLAUSE[tool_name]
    for sandboxed, full_access in _FULL_ACCESS_SUBSTITUTIONS:
        description = description.replace(sandboxed, full_access.format(clause = clause))
    return description


def _build_terminal_shell_note() -> str:
    """Shell-specific note, on the TERMINAL description only.

    Which shell runs is read from the resolver, not assumed: telling a model it has bash on a host
    where _get_shell_cmd fell back to cmd reintroduces the multi-line half-execution this note
    exists to prevent. It stays off the python description because none of it applies to the python
    sandbox, and naming a shell there invites subprocess/os.system as a way past the terminal
    blocklist. No program is named either: powershell/pwsh are on that blocklist and `cmd /c start`
    is not, so recommending one hands back a hard block and recommending the other advertises the
    gap.
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

# Separate full-access schemas; the sandboxed pair stays the module default.
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

# The isolated Terminal is cmd.exe inside MXC whatever the host shell, so it has its own schema.
_ISOLATED_CMD_SHELL_NOTE = (
    " The shell is cmd, running isolated, not bash: send one command per call, chain with &&, use "
    "double quotes only, and use relative paths. git, when installed, runs without hooks, a pager or "
    "an editor, so pass -m to git commit."
)

TERMINAL_TOOL_CMD_ISOLATED = {
    "type": "function",
    "function": {
        **TERMINAL_TOOL["function"],
        "description": "Execute a terminal command and return stdout/stderr."
        + _SANDBOX_PATHS_NOTE
        + _ISOLATED_CMD_SHELL_NOTE,
    },
}


def apply_terminal_profile_description(tools: list[dict], profile: str) -> list[dict]:
    """Swap the terminal schema for the one matching ``profile``. Like
    apply_full_access_tool_descriptions, the input list is never mutated and a list with nothing to
    swap is returned as-is; only "cmd_isolated" changes anything."""
    if not tools or profile != "cmd_isolated":
        return tools
    swapped = False
    out: list[dict] = []
    for tool in tools:
        name = (tool.get("function") or {}).get("name") if isinstance(tool, dict) else None
        if name == "terminal":
            out.append(TERMINAL_TOOL_CMD_ISOLATED)
            swapped = True
        else:
            out.append(tool)
    return out if swapped else tools


_FULL_ACCESS_TOOL_BY_NAME = {
    "python": PYTHON_TOOL_FULL_ACCESS,
    "terminal": TERMINAL_TOOL_FULL_ACCESS,
}


def apply_full_access_tool_descriptions(tools: list[dict]) -> list[dict]:
    """Swap python/terminal/edit_file for their Full access schemas. Only the sandboxed built-ins
    are touched; web_search, render_html, search_knowledge_base and MCP tools are passed through
    untouched, and a list without any of them is returned as-is so callers can apply this
    unconditionally. The input list is never mutated: ALL_TOOLS entries are module globals shared
    across requests."""
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
        # The description steers models off heredocs.
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

# Appended: otherwise a model that thinks it cannot reach a checkout rewrites whole files.
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

# Duplicated so the registry never imports the RAG stack.
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
READ_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "read_skill",
        "description": (
            "Read instructions or a UTF-8 resource from an enabled Agent Skill. "
            "Start with SKILL.md, then read referenced resources only when needed. "
            "This tool reads files; it does not execute scripts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Enabled skill name from the system skill catalog.",
                },
                "resource": {
                    "type": "string",
                    "description": "Relative resource path. Defaults to SKILL.md.",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Character offset for the next page. Defaults to 0.",
                },
            },
            "required": ["name"],
        },
    },
}
CREATE_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "create_skill",
        "description": (
            "Create a new Agent Skill in ~/.agents/skills. Use the skill-creator instructions "
            "first. Existing skills are never overwritten."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Lowercase skill name using letters, numbers, and single hyphens.",
                },
                "description": {
                    "type": "string",
                    "description": "When the skill should be used, in 1-1024 characters.",
                },
                "instructions": {
                    "type": "string",
                    "description": "Complete Markdown instructions for the skill body.",
                },
            },
            "required": ["name", "description", "instructions"],
        },
    },
}


from .view_image import VIEW_IMAGE_TOOL


ALL_TOOLS = [
    WEB_SEARCH_TOOL,
    PYTHON_TOOL,
    TERMINAL_TOOL,
    EDIT_FILE_TOOL,
    VIEW_IMAGE_TOOL,
    RENDER_HTML_TOOL,
    SEARCH_KNOWLEDGE_BASE_TOOL,
    SEARCH_CONVERSATION_TOOL,
]

# An ordinary tool so all three tool loops behave the same; never in ALL_TOOLS. The client keys
# the handoff on this opening, since denied/skipped calls also end with tool_end.
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


# OpenAI's function.name regex; MCP names that violate it would 400 the whole request, so they ship under an alias.
_OPENAI_FN_NAME_MAX = 64
_OPENAI_FN_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,%d}$" % _OPENAI_FN_NAME_MAX)

_MCP_ALIAS_DIGEST_LEN = 8
_MCP_ALIAS_SUFFIX_LEN = _MCP_ALIAS_DIGEST_LEN + 1

_MCP_COMPACT_SPEC_CHARS = 1500
_MCP_SUMMARY_CHARS = 240
_MCP_COMPACT_HINT = "Full parameters via mcp_tool_schema."
_MCP_MIN_SCHEMA_PAGE_CHARS = 64
_MCP_FULL_LISTING_SHARE = 0.75
_MCP_LISTING_CONTEXT_TOKENS: ContextVar = ContextVar("mcp_listing_context_tokens", default = None)
_MCP_COMPACTED_WINDOWS: dict[tuple, frozenset] = {}


def set_mcp_listing_context_tokens(context_tokens) -> None:
    """The local window the next MCP listing in this context is sized against; unset lists every tool in full."""
    valid = isinstance(context_tokens, int) and context_tokens > 0
    _MCP_LISTING_CONTEXT_TOKENS.set(context_tokens if valid else None)


MCP_TOOL_SCHEMA_TOOL = {
    "type": "function",
    "function": {
        "name": "mcp_tool_schema",
        "description": (
            "Return the full description and parameter schema of an MCP tool. A tool whose "
            f"listing ends with '{_MCP_COMPACT_HINT}' shows only its top-level parameters; "
            "call this before using it when that listing is not enough."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The MCP tool name exactly as listed, including its mcp__ prefix.",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Character offset for the next page. Defaults to 0.",
                },
            },
            "required": ["name"],
        },
    },
}


def _mcp_input_schema(tool: dict) -> dict:
    return (
        tool.get("inputSchema") or tool.get("input_schema") or {"type": "object", "properties": {}}
    )


def _mcp_spec_compacted(tool: dict) -> bool:
    schema_chars = len(json.dumps(_mcp_input_schema(tool), separators = (",", ":")))
    return schema_chars + len(tool.get("description") or "") > _MCP_COMPACT_SPEC_CHARS


def _mcp_summary(description: str) -> str:
    text = " ".join((description or "").split()).lstrip("# ")
    match = re.match(r"(.+?[.!?])(?:\s|$)", text)
    if match:
        text = match.group(1)
    if len(text) > _MCP_SUMMARY_CHARS:
        text = text[: _MCP_SUMMARY_CHARS - 3].rstrip() + "..."
    return text


def _mcp_compact_parameters(schema: dict) -> dict:
    properties: dict[str, dict] = {}
    for key, value in (schema.get("properties") or {}).items():
        prop: dict = {}
        if isinstance(value, dict):
            branches = value.get("anyOf") or value.get("oneOf") or []
            branch_types = [b.get("type") for b in branches if isinstance(b, dict)]
            if isinstance(value.get("type"), (str, list)):
                prop["type"] = value["type"]
            elif branch_types and all(isinstance(kind, str) for kind in branch_types):
                kinds = list(dict.fromkeys(branch_types))
                prop["type"] = kinds[0] if len(kinds) == 1 else kinds
            if isinstance(value.get("enum"), list) and len(json.dumps(value["enum"])) <= 200:
                prop["enum"] = value["enum"]
        # llama.cpp compiles an empty schema to an object-only grammar; a description alone accepts any value.
        properties[key] = prop or {"description": "See mcp_tool_schema."}
    compact: dict = (
        {"type": "object", "properties": properties} if properties else {"type": "object"}
    )
    if isinstance(schema.get("required"), list):
        compact["required"] = schema["required"]
    return compact


def _mcp_compact_spec(name: str, display: str, tool: dict, description: str) -> dict:
    parts = (f"[{display}]", _mcp_summary(description), _MCP_COMPACT_HINT)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": " ".join(part for part in parts if part),
            "parameters": _mcp_compact_parameters(_mcp_input_schema(tool)),
        },
    }


def _mcp_tool_schema_text(display: str, tool: dict) -> str:
    schema = json.dumps(_mcp_input_schema(tool), separators = (",", ":"))
    description = " ".join((tool.get("description") or "").split())
    return f"[{display}] {tool.get('name')}: {description}\n\nParameters (JSON Schema): {schema}"


def _mcp_cached_tool(server: dict, tool_name: str) -> dict | None:
    for tool in get_cached_tools(server["id"]) or []:
        if tool.get("name") == tool_name and tool_visible_to(tool, "model"):
            return public_tool(server, tool)
    return None


def _mcp_image_recipient(server: dict, mapping: dict) -> str:
    identity = [
        server["id"],
        server["url"],
        server.get("headers_json"),
        server.get("use_oauth"),
        mapping,
    ]
    return hashlib.sha256(json.dumps(identity, sort_keys = True).encode()).hexdigest()


def _mcp_image_destination(url: str) -> str:
    # Host or program name only: credentials can sit in URL userinfo or in stdio arguments.
    if is_stdio(url):
        try:
            return f"local command {os.path.basename(parse_stdio_command(url)[0])}"
        except (ValueError, IndexError):
            return "local command"
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or "unknown host"
    return f"{host}:{parts.port}" if parts.port else host


def mcp_image_share(name, arguments, mcp_image) -> dict | None:
    """Approval-card details plus the image bound to this server when the call would send it, else None.

    A call that would send it has ``arguments`` rewritten in place to what goes out (settle_image_call).
    """
    if mcp_image is None or not isinstance(arguments, dict):
        return None
    from .tool_loop_controller import UNPARSED_ARGUMENTS_KEY  # noqa: PLC0415

    server, tool, tool_name = _mcp_resolve_tool(name)
    mapping = image_mapping(server, tool) if server else None
    if mapping is None:
        return None
    # Arguments that could not be read are not a call to rewrite: they keep their ordinary path.
    schema = _mcp_input_schema(tool)
    properties = schema.get("properties") or {}
    if UNPARSED_ARGUMENTS_KEY in arguments or (
        set(arguments) == {"raw"} and "raw" not in properties
    ):
        return None
    if not settle_image_call(arguments, mapping["field"], schema.get("required") or ()):
        return None
    return {
        "disclosure": {
            "server": server.get("display_name") or server["id"],
            "tool": tool_name,
            "size_bytes": len(mcp_image.data),
            "destination": _mcp_image_destination(server["url"]),
        },
        "image": mcp_image.approved_for(_mcp_image_recipient(server, mapping)),
    }


def _mcp_resolve_tool(name) -> "tuple[dict | None, dict | None, str]":
    if not isinstance(name, str) or not name.startswith(MCP_TOOL_PREFIX) or name.count("__") < 2:
        return None, None, ""
    _, server_key, _ = name.split("__", 2)
    tool_name = _mcp_raw_tool_name(name)
    server = mcp_servers_db.get_server_for_tool(server_key)
    return server, _mcp_cached_tool(server, tool_name) if server else None, tool_name


def mcp_image_targets(names) -> list[tuple[str, str]]:
    """(catalog name, field) for each of these tools with a field mapped to the attached image."""
    targets = []
    for name in names:
        server, tool, _ = _mcp_resolve_tool(name)
        mapping = image_mapping(server, tool) if server else None
        if mapping:
            targets.append((name, mapping["field"]))
    return targets


def mcp_catalog_takes_image(names) -> bool:
    """Whether any of these catalog tools has a field mapped to the attached image."""
    for name in names:
        server, tool, _ = _mcp_resolve_tool(name)
        if server and image_mapping(server, tool):
            return True
    return False


def mcp_tool_input_schema(name) -> dict | None:
    tool = _mcp_resolve_tool(name)[1]
    return _mcp_input_schema(tool) if tool is not None else None


def _mcp_schema_page(prefix: str, text: str, offset: int) -> str:
    page_chars = _tool_result_char_budget()
    while True:
        end = min(offset + page_chars, len(text))
        page = prefix + text[offset:end]
        if end < len(text):
            page += (
                f"\n\n[Characters {offset}-{end} of {len(text)}. "
                f"Call mcp_tool_schema with offset={end} for the rest.]"
            )
        if _fit_result_to_room(page, "mcp_tool_schema") == page:
            return page
        if page_chars < _MCP_MIN_SCHEMA_PAGE_CHARS:
            return _fit_result_to_room(
                (prefix or "Error: ") + "Not enough context room to read this MCP tool schema. "
                "Reduce the conversation context and retry.",
                "mcp_tool_schema",
            )
        page_chars //= 2


def _mcp_tool_schema(name, offset = None) -> str:
    server, tool, tool_name = _mcp_resolve_tool(name)
    if not tool_name:
        return "Error: mcp_tool_schema needs an MCP tool name as listed, such as mcp__<server>__<tool>."
    if not server:
        return f"Error: MCP server for tool '{tool_name}' not found"
    display = server.get("display_name") or server["id"]
    if tool is None:
        return f"Error: MCP server '{display}' does not list a tool named '{tool_name}'"
    text = _mcp_tool_schema_text(display, tool)
    offset = 0 if offset is None else offset
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset < len(text):
        return f"Error: offset must be an integer from 0 to {len(text) - 1}."
    return _mcp_schema_page("", text, offset)


def _mcp_compact_candidates(listed) -> list[tuple[int, dict]]:
    """(index in the flat listing, compact spec) for every large tool, largest saving first."""
    candidates: list[tuple[int, int, dict]] = []
    index = 0
    for server, payload, server_specs in listed:
        display = server.get("display_name") or server["id"]
        by_name = {tool.get("name"): tool for tool in payload if isinstance(tool, dict)}
        for spec in server_specs:
            function = spec["function"]
            tool = by_name.get(_mcp_raw_tool_name(function["name"]))
            if tool is not None and _mcp_spec_compacted(tool):
                description = function["description"].removeprefix(f"[{display}]").strip()
                compact = _mcp_compact_spec(function["name"], display, tool, description)
                saving = len(json.dumps(spec, separators = (",", ":"))) - len(
                    json.dumps(compact, separators = (",", ":"))
                )
                if saving > 0:
                    candidates.append((saving, index, compact))
            index += 1
    candidates.sort(key = lambda item: -item[0])
    return [(index, compact) for _, index, compact in candidates]


def _mcp_listing(listed: list[tuple[dict, list[dict], list[dict]]]) -> list[dict]:
    specs = [spec for _, _, server_specs in listed for spec in server_specs]
    ctx = _MCP_LISTING_CONTEXT_TOKENS.get()
    if not ctx or not specs:
        return specs
    budget = ctx * _MCP_FULL_LISTING_SHARE
    text = json.dumps(specs, separators = (",", ":"))
    listing_tokens = _text_token_cost(text, ctx)
    if listing_tokens <= budget:
        _MCP_COMPACTED_WINDOWS[(current_account_id(), ctx)] = frozenset()
        return specs
    # Compact the largest first and stop once it fits; dropping nested params costs accuracy.
    tokens_per_char = listing_tokens / max(len(text), 1)
    budget -= _text_token_cost(json.dumps(MCP_TOOL_SCHEMA_TOOL, separators = (",", ":")), ctx)
    listing = list(specs)
    candidates = _mcp_compact_candidates(listed)
    chars = len(text)
    compacted: set[str] = set()
    for position, (index, compact) in enumerate(candidates):
        chars -= len(json.dumps(listing[index], separators = (",", ":"))) - len(
            json.dumps(compact, separators = (",", ":"))
        )
        listing[index] = compact
        compacted.add(compact["function"]["name"])
        if chars * tokens_per_char > budget:
            continue
        if (
            position == len(candidates) - 1
            or _text_token_cost(json.dumps(listing, separators = (",", ":")), ctx) <= budget
        ):
            break
    _MCP_COMPACTED_WINDOWS[(current_account_id(), ctx)] = frozenset(compacted)
    if compacted:
        listing.append(MCP_TOOL_SCHEMA_TOOL)
    return listing


def _mcp_listing_compacted(name: str) -> bool:
    key = (current_account_id(), _window_context_tokens() or 0)
    return name in _MCP_COMPACTED_WINDOWS.get(key, frozenset())


def _mcp_tool_names(server: dict, mcp_tools: list[dict]) -> dict[str, str]:
    """Composed function name -> raw MCP name, for the tools this server ships to a model.

    Names that already satisfy ``function.name`` are claimed first, so an alias minted for a dotted
    or oversized one can never take a name another tool ships under.
    """
    server_key = "blender" if server.get("builtin_id") == "blender" else server["id"]
    prefix = f"{MCP_TOOL_PREFIX}{server_key}__"
    raw_names = [
        tool["name"] for tool in mcp_tools if tool.get("name") and tool_visible_to(tool, "model")
    ]
    names: dict[str, str] = {}
    for raw_name in raw_names:
        if _OPENAI_FN_NAME_RE.fullmatch(prefix + raw_name):
            names.setdefault(prefix + raw_name, raw_name)
    stem_room = max(0, _OPENAI_FN_NAME_MAX - len(prefix) - _MCP_ALIAS_SUFFIX_LEN)
    for raw_name in raw_names:
        if _OPENAI_FN_NAME_RE.fullmatch(prefix + raw_name):
            continue
        encoded = raw_name.encode("utf-8", "surrogatepass")
        digest = hashlib.sha256(encoded).hexdigest()[:_MCP_ALIAS_DIGEST_LEN]
        stem = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name)[:stem_room]
        alias = f"{prefix}{stem}_{digest}"
        if _OPENAI_FN_NAME_RE.fullmatch(alias):
            names.setdefault(alias, raw_name)
    return names


# Kept apart from the tool cache so an in-flight call still resolves after eviction.
_MCP_TOOL_ALIASES: dict[str, str] = {}


def _mcp_raw_tool_name(name: str) -> str:
    return _MCP_TOOL_ALIASES.get(name) or name.split("__", 2)[-1]


def _mcp_specs_for_server(server: dict, mcp_tools: list[dict]) -> list[dict]:
    """Convert an MCP server's tool list into OpenAI function specs."""
    display = server.get("display_name") or server["id"]
    names_by_raw = {raw_name: name for name, raw_name in _mcp_tool_names(server, mcp_tools).items()}
    specs: list[dict] = []
    seen_names: set[str] = set()
    for tool in mcp_tools:
        raw_name = tool.get("name") or ""
        if not raw_name:
            logger.warning("Skipping MCP tool on '%s': empty name.", display)
            continue
        if not tool_visible_to(tool, "model"):
            logger.debug("Skipping app-only MCP tool '%s' on '%s'.", raw_name, display)
            continue
        name = names_by_raw.get(raw_name)
        if name is None:
            logger.warning(
                "Skipping MCP tool '%s' on '%s': no free OpenAI function.name for it.",
                raw_name,
                display,
            )
            continue
        if name in seen_names:
            logger.warning("Skipping duplicate MCP tool '%s' on '%s'.", raw_name, display)
            continue
        seen_names.add(name)
        description = tool.get("description") or ""
        if name.split("__", 2)[2] != raw_name:
            _MCP_TOOL_ALIASES[name] = raw_name
            # The alias is the only name the model may emit; the description carries the real name.
            description = f"({raw_name}) {description}"
        else:
            _MCP_TOOL_ALIASES.pop(name, None)
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"[{display}] {description}".strip(),
                    # mcp<2 dumps "inputSchema", 2.x "input_schema"; accept both.
                    "parameters": tool.get("inputSchema")
                    or tool.get("input_schema")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return specs


def _enabled_mcp_servers(servers: list[dict]) -> list[dict]:
    enabled = [server for server in servers if server.get("is_enabled")]
    if not any(is_studio_decisions(server["url"]) for server in enabled):
        return enabled
    from utils import systemone_settings

    return (
        enabled
        if systemone_settings.get_enabled()
        else [server for server in enabled if not is_studio_decisions(server["url"])]
    )


def cached_mcp_tools() -> tuple[list[dict], bool]:
    """The MCP schemas already in cache, and whether that is the whole set.

    get_enabled_mcp_tools() renders the same servers, but reaches the network to do it: it spawns
    stdio servers, blocks for a probe timeout on one that is down, and writes cache and cool-off
    state. A background token count must not do any of that, so this reads only what those probes
    have already filled in.

    ``complete`` is False when an enabled server has nothing cached and is not in its post-failure
    cool-off, meaning a completion would probe it and render schemas this cannot price. A cool-off
    server renders nothing on the completion path either, so skipping that one is exact rather than
    short. Callers that must not undercount should decline on False.
    """
    servers = _enabled_mcp_servers(mcp_servers_db.list_servers())
    if not stdio_mcp_enabled():
        servers = [s for s in servers if not is_stdio(s["url"])]

    listed: list[tuple[dict, list[dict], list[dict]]] = []
    complete = True
    for server in servers:
        payload = get_cached_tools(server["id"])
        if payload is None:
            if not in_failure_cooloff(server["id"]):
                complete = False
            continue
        payload = [public_tool(server, tool) for tool in payload]
        listed.append((server, payload, _mcp_specs_for_server(server, payload)))
    return _mcp_listing(listed), complete


async def get_enabled_mcp_tools(
    include_stdio: bool = True, server_ids: set[str] | None = None
) -> list[dict]:
    # keep the SQLite-backed server list off the event loop.
    servers = await asyncio.to_thread(lambda: _enabled_mcp_servers(mcp_servers_db.list_servers()))
    if server_ids is not None:
        servers = [server for server in servers if server["id"] in server_ids]
    if not include_stdio or not stdio_mcp_enabled():
        servers = [s for s in servers if not is_stdio(s["url"])]
    if not servers:
        return []

    # cool-off avoids blocking every send for the full probe timeout when a server is down.
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
                    **oauth_client_kwargs(s),
                )
                for s in uncached
            ),
            return_exceptions = True,
        )
        # On-loop re-read so an edit cannot invalidate between it and the cache writes.
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
                record_probe_failure(server["id"], bool(fresh.get("use_oauth")))
                continue
            cache_tools(server["id"], payload)

    listed: list[tuple[dict, list[dict], list[dict]]] = []
    for server in servers:
        payload = get_cached_tools(server["id"])
        if payload is None:
            continue
        payload = [public_tool(server, tool) for tool in payload]
        listed.append((server, payload, _mcp_specs_for_server(server, payload)))
    return _mcp_listing(listed)


def mcp_search_argument(name: str, tool: dict) -> str | None:
    schema = _mcp_input_schema(tool)
    required = schema.get("required") or []
    properties = schema.get("properties") or {}
    if len(required) != 1 or not isinstance(properties, dict):
        return None
    key = required[0]
    prop = properties.get(key) if isinstance(key, str) else None
    if not isinstance(prop, dict) or prop.get("type") != "string" or "enum" in prop:
        return None
    if is_potentially_unsafe_tool_call(name, {key: ""}):
        return None
    return key


async def mcp_search_tools(
    include_stdio: bool = True, server_ids: set[str] | None = None
) -> list[dict]:
    from state.tool_policy import get_tool_policy

    if get_tool_policy() is False:
        return []
    await get_enabled_mcp_tools(include_stdio = include_stdio, server_ids = server_ids)
    servers = _enabled_mcp_servers(await asyncio.to_thread(mcp_servers_db.list_servers))
    if server_ids is not None:
        servers = [server for server in servers if server["id"] in server_ids]
    if not include_stdio or not stdio_mcp_enabled():
        servers = [s for s in servers if not is_stdio(s["url"])]
    found = []
    for server in servers:
        for tool in get_cached_tools(server["id"]) or ():
            raw_name = tool.get("name") if isinstance(tool, dict) else None
            if not isinstance(raw_name, str) or not tool_visible_to(tool, "model"):
                continue
            name = f"{MCP_TOOL_PREFIX}{server['id']}__{raw_name}"
            argument = mcp_search_argument(name, public_tool(server, tool))
            if argument:
                found.append(
                    {
                        "name": name,
                        "serverId": server["id"],
                        "serverName": server.get("display_name") or server["id"],
                        "tool": raw_name,
                        "description": tool.get("description") or "",
                        "argument": argument,
                    }
                )
    return found


def execute_mcp_tool(name: str, arguments: dict, **kwargs) -> str:
    if not name.startswith(MCP_TOOL_PREFIX):
        return f"Error: '{name}' is not an MCP tool"
    return execute_tool(name, arguments, **kwargs)


def mcp_tool_definition(server_id: str, tool_name: str) -> "dict | None":
    """cache only: callers must not spawn a stdio subprocess or block on a probe."""
    tools = get_cached_tools(server_id) or ()
    return next((t for t in tools if isinstance(t, dict) and t.get("name") == tool_name), None)


def mcp_session_scope(session_id: "str | None", thread_id: "str | None") -> "str | None":
    """Persist a stateful stdio session only per conversation (thread_id). session_id is the project-wide sandbox
    id, so scoping by it alone leaks browser/DB/REPL state across conversations; fall back to one-shot. Tag +
    percent-quote the parts so ids can't collide or ":" merge conversations."""
    if not thread_id:
        return None
    quote = urllib.parse.quote
    return f"s={quote(session_id or '', safe = '')}:t={quote(thread_id, safe = '')}"


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
    *,
    tool_execution_mode: str = "auto",
    host_access_approved: bool = False,
    mcp_image = None,
) -> str:
    """Execute a tool by name with the given arguments; returns a string.

    ``timeout``: int seconds, ``None`` = no limit, unset = ``_EXEC_TIMEOUT``. ``session_id``:
    optional ID for per-conversation sandbox isolation. ``thread_id``: optional conversation ID;
    scopes stateful MCP stdio sessions per thread (session_id alone can be shared project-wide).
    ``rag_scope``: hidden per-request RAG context the model never sees; consumed by
    ``search_knowledge_base``. ``disable_sandbox``: Bypass Permissions; run python/terminal without
    the safety checks, blocklist, or resource caps (secrets still stripped). Only affects local code
    tools; web_search / MCP are unchanged. ``output_callback``: optional ``callable(str)`` invoked
    with incremental stdout/stderr chunks while python/terminal executions run. Purely
    observational: the returned result string is identical with or without it. ``website_policy``:
    hidden server-validated domain limits for web_search. ``tool_execution_mode`` controls OS
    isolation for python/terminal: ``"auto"`` isolates when available and otherwise preserves
    existing behavior, while ``"required"`` refuses unisolated execution. Full access remains
    controlled by ``disable_sandbox``; ``"full"`` here is refused. ``host_access_approved``: the user approved this
    call at the confirmation prompt (see ``_prepare_tool_launch``).
    """
    from state.tool_policy import require_tool_access

    require_tool_access(disable_sandbox = disable_sandbox)
    logger.info(f"execute_tool: name={name}, session_id={session_id}, timeout={timeout}")
    # Set unconditionally so a stale value from this thread is never read; no reset needed.
    _REQUEST_CONTEXT_TOKENS.set(context_tokens)
    _REQUEST_RESULT_BUDGET.set(result_budget_tokens)
    # Answered here: the tool would blame the model for keys a truncated call never sent.
    # Imported locally to avoid an import cycle.
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
    # Block placeholders copied from compacted history before any tool can write them (#11839).
    receipt_field = compaction_receipt_field(
        arguments,
        match_only = frozenset({"old_string"}) if name == "edit_file" else frozenset(),
    )
    if receipt_field is not None:
        return (
            f"Error: {name} '{receipt_field}' is a placeholder that stood in for earlier "
            "arguments to save room, not content, so nothing ran. Write the actual content "
            "out in full."
        )
    # By type, not `is _TIMEOUT_UNSET`: see `_request_context_tokens`.
    effective_timeout = (
        timeout if timeout is None or isinstance(timeout, (int, float)) else _EXEC_TIMEOUT
    )
    if name == "create_skill":
        from .skills import SkillError, create_skill

        try:
            record = create_skill(
                arguments.get("name", ""),
                arguments.get("description", ""),
                arguments.get("instructions", ""),
            )
        except SkillError as exc:
            return f"Error: {exc}"
        from routes.inference import _invalidate_agent_skills_cache

        _invalidate_agent_skills_cache()
        return (
            f"Created Agent Skill '{record['name']}' at {record['path']}. "
            f"It is enabled and ready to use."
        )

    if name == "read_skill":
        from .skills import (
            MAX_SKILL_PAGE_CHARS,
            MIN_SKILL_PAGE_CHARS,
            SkillError,
            read_skill_resource,
        )
        try:
            page_chars = MAX_SKILL_PAGE_CHARS
            resource = arguments.get("resource")
            offset = arguments.get("offset")
            while True:
                result = read_skill_resource(
                    arguments.get("name") or "",
                    "SKILL.md" if resource is None else resource,
                    0 if offset is None else offset,
                    page_chars = page_chars,
                )
                fitted = _fit_result_to_room(result, name)
                if fitted == result:
                    return result
                if page_chars < MIN_SKILL_PAGE_CHARS:
                    return (
                        "Error: Not enough context room to read this skill resource. "
                        "Reduce the conversation context and retry the same read_skill call."
                    )
                page_chars //= 2
        except SkillError as exc:
            return f"Error: {exc}"

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
    if name == "mcp_tool_schema":
        return _mcp_tool_schema(arguments.get("name"), arguments.get("offset"))
    if name.startswith(MCP_TOOL_PREFIX):
        # An MCP server is not inside the terminal sandbox, so the local refusal has to hold here too.
        if _mcp_arguments_reference_studio_credential(arguments):
            return _STUDIO_CREDENTIAL_BLOCKED
        try:
            _, server_id, _ = name.split("__", 2)
        except ValueError:
            return f"Error: malformed MCP tool name '{name}'"
        tool_name = _mcp_raw_tool_name(name)
        server = mcp_servers_db.get_server_for_tool(server_id)
        if not server:
            return f"Error: MCP server for tool '{tool_name}' not found"
        server_id = server["id"]
        display = server.get("display_name") or server_id
        if not server.get("is_enabled"):
            return f"Error: MCP server '{display}' is disabled"
        if is_stdio(server["url"]) and not stdio_mcp_enabled():
            return f"Error: stdio MCP server '{display}' is disabled on this host"
        tool = _mcp_cached_tool(server, tool_name) if _mcp_listing_compacted(name) else None
        if tool is not None and isinstance(arguments, dict):
            missing = [
                key for key in _mcp_input_schema(tool).get("required") or [] if key not in arguments
            ]
            if missing:
                return _mcp_schema_page(
                    f"Error: MCP tool '{tool_name}' requires {', '.join(missing)}.\n\n",
                    _mcp_tool_schema_text(display, tool),
                    0,
                )
        mcp_scope = mcp_session_scope(session_id, thread_id)
        headers = parse_server_headers(server)
        url = server["url"]
        use_oauth = bool(server.get("use_oauth"))
        mapping = (
            image_mapping(server, tool or _mcp_cached_tool(server, tool_name))
            if image_input_mappings(server) and isinstance(arguments, dict)
            else None
        )
        carries_image = bool(mapping) and arguments.get(mapping["field"]) == ATTACHED_IMAGE
        if mcp_image is not None and not carries_image:
            return (
                "Error: the MCP server changed after the image was approved. Call the tool again."
            )
        if carries_image:
            if mcp_image is None:
                return "Error: no approved image to send. Ask the user to attach one and approve sharing it."
            # Re-read the row: an edit while the approval card was open must not redirect the image.
            fresh = mcp_servers_db.get_server(server_id)
            fresh_mapping = (
                image_mapping(fresh, tool or _mcp_cached_tool(fresh, tool_name)) if fresh else None
            )
            if not (
                fresh_mapping
                and fresh.get("is_enabled")
                and mcp_image.recipient
                == _mcp_image_recipient(server, mapping)
                == _mcp_image_recipient(fresh, fresh_mapping)
            ):
                return "Error: the MCP server changed after the image was approved. Call the tool again."
            arguments = {**arguments, mapping["field"]: mcp_image.encoded(mapping["encoding"])}

        def _image_still_approved(row: dict) -> bool:
            if not carries_image:
                return True
            current = image_mapping(row, tool or _mcp_cached_tool(row, tool_name))
            return bool(current) and _mcp_image_recipient(row, current) == mcp_image.recipient

        def _config_current() -> bool:
            # Re-read before caching a session: an update/delete (or OAuth switch) may have raced this call.
            row = mcp_servers_db.get_server(server_id)
            return (
                row is not None
                and bool(row.get("is_enabled"))
                and row.get("url") == url
                and parse_server_headers(row) == headers
                and bool(row.get("use_oauth")) == use_oauth
                and _image_still_approved(row)
            )

        result = call_tool_sync(
            url = url,
            headers = headers,
            name = tool_name,
            args = arguments,
            timeout = effective_timeout,
            use_oauth = use_oauth,
            cancel_event = cancel_event,
            scope = mcp_scope,
            **oauth_client_kwargs(server),
            config_check = _config_current,
            ui_resource_uri = tool_ui_resource_uri(mcp_tool_definition(server_id, tool_name)),
        )
        if mcp_image is not None and isinstance(result, str):
            result, returned_images, _ = result.partition(MCP_IMAGES_SENTINEL)
            if returned_images:
                result = (
                    result.rstrip("\n")
                    + "\n[Images the tool returned were withheld from the model.]"
                )
            result = mcp_image.redact(result)
        if tool is not None and isinstance(result, str) and result.startswith("Error:"):
            return _mcp_schema_page(
                result.rstrip() + "\n\n", _mcp_tool_schema_text(display, tool), 0
            )
        return _fit_result_to_room(result, name)
    if name == "deep_research":
        if not str(arguments.get("question") or "").strip():
            return "Error: deep_research needs a question to investigate."
        return DEEP_RESEARCH_STARTED
    if name == "web_search":
        query, url = _resolve_web_search_args(arguments)
        image_queries = arguments.get("image_queries") if isinstance(arguments, dict) else None
        if not query and not url and not _clean_image_queries(image_queries):
            return "No query provided."
        return _fit_result_to_room(
            _web_search(
                query,
                url = url or None,
                timeout = effective_timeout,
                cancel_event = cancel_event,
                website_policy = website_policy,
                include_images = search_images,
                image_queries = image_queries,
            ),
            name,
        )
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
                tool_execution_mode = tool_execution_mode,
                host_access_approved = host_access_approved,
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
                tool_execution_mode = tool_execution_mode,
                host_access_approved = host_access_approved,
            )
    if name == "view_image":
        from .view_image import view_image
        with _session_in_flight(session_id):
            return _fit_result_to_room(
                view_image(arguments.get("path"), _get_workdir(session_id), cancel_event), name
            )
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
    """Run the RAG search bound to the hidden per-request ``rag_scope`` (the model supplies only
    ``query``/``top_k``). Lazy import; missing sqlite-vec degrades to a friendly message."""
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
    if sources:
        import json as _json
        return text + RAG_SOURCES_SENTINEL + _json.dumps(sources, ensure_ascii = False)
    return text


# Small: whole archived turns land in a protected exchange the window cannot trim.
_MAX_CONVERSATION_SEARCH_TOP_K = 8


def _search_conversation(arguments: dict, rag_scope: dict | None) -> str:
    """Search this thread's archived turns. ``rag_scope`` carries only the thread id here; the model
    supplies ``query``/``top_k``."""
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

    # Clamped: a negative top_k slices as out[:-1], nearly the whole pool.
    requested = _opt_int((arguments or {}).get("top_k"))
    # Omitted top_k falls through to the configured default, not the ceiling.
    top_k = (
        None if requested is None else max(1, min(_MAX_CONVERSATION_SEARCH_TOP_K, int(requested)))
    )
    # Also against the room left: the fixed cap bounds the ask, not what fits.
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
        # Room caps the default; it is not a target.
        top_k = (
            min(default_k, _MAX_CONVERSATION_SEARCH_TOP_K, affordable)
            if top_k is None
            else max(1, min(top_k, affordable))
        )

    # This request's branch, so a response replaced by Retry is not searchable.
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

    # Halve until the rendered result fits: chunk size is a target, not a weight (overlap, other
    # tokenizer, markup). A single chunk that still does not fit is refused.
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


# Role, call id and template wrapping around a tool message.
_TOOL_MESSAGE_FRAMING_TOKENS = 8


def _conversation_search_cost(text: str, counter = None) -> int:
    """What admitting this result really costs, exactly when the caller has a tokenizer. The
    estimate below is pessimistic for CJK and emoji but still optimistic for ASCII that tokenises
    densely (source code, minified JSON, hashes, command output all run nearer two or three
    characters per token than four), so a result could be admitted at well under its real cost
    and then land in the current tool exchange, which the window is not allowed to evict. A
    tokenizer-backed caller passes its own counter, and the GGUF path is one."""
    if counter is not None:
        try:
            return int(counter(text)) + _TOOL_MESSAGE_FRAMING_TOKENS
        except Exception:
            logger.debug("conversation search: exact result count failed", exc_info = True)
    return _conversation_search_tokens(text) + _TOOL_MESSAGE_FRAMING_TOKENS


def _conversation_search_tokens(text: str) -> int:
    """A deliberately pessimistic size for a search result, in tokens. The shared estimator charges
    four characters per token, which is about right for English and badly wrong for text that
    tokenises densely: CJK and emoji run closer to one token per character, so a result could be
    accepted at a quarter of its real cost and then land in the current tool exchange, which the
    window cannot evict. No exact counter is reachable from here, the provider loop having no
    tokenizer at all, so non-ASCII characters are charged one token each and the rest at the
    usual rate."""
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
    """Admission-controlled RAG search. ``search_fn`` swaps in a different search over the same
    capacity-of-one slot, so archive lookups queue behind document lookups instead of racing for
    the embedder."""
    search_fn = search_fn or _search_knowledge_base
    if cancel_event is not None and cancel_event.is_set():
        return "Error: knowledge base search cancelled."
    deadline = time.monotonic() + timeout if timeout is not None else None
    while not _RAG_SEARCH_SLOT.acquire(timeout = 0.05):
        if cancel_event is not None and cancel_event.is_set():
            return "Error: knowledge base search cancelled."
        if deadline is not None and time.monotonic() >= deadline:
            return "Error: knowledge base search timed out."

    # The worker releases the slot in its finally, not the caller on timeout, so concurrency stays one.
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
        account_thread(target = search, name = "rag-tool-search", daemon = True).start()
    except Exception:
        release_slot()
        raise
    while True:
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


# A high floor keeps forced retrieval precise; tunable via RAG_AUTOINJECT_MIN_SCORE.
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
    """Whether a thread-attached file should be injected in full rather than retrieved top-K.
    ``rag_scope.whole_doc=False`` disables it for this request."""
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
        total += 4
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
            return strip_current_date_update_note(strip_attached_image_note(content)).strip()
        if isinstance(content, list):
            parts = [
                strip_attached_image_note(p.get("text", ""))
                for p in content
                if isinstance(p, dict) and p.get("type") in ("text", "input_text")
            ]
            return strip_current_date_update_note(" ".join(t for t in parts if t)).strip()
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
    """Render a retrieval the loop never asked for as a normal tool exchange. Returns ``{"events":
    [...], "messages": [...]}``: the messages are what the model reads, the events what the UI
    draws, so a forced retrieval shows up as an ordinary tool card with working citations instead
    of appearing from nowhere."""
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

    Deliberately NOT gated on ``rag_scope``: compaction happens whether or not document RAG is on,
    and these turns are the conversation's own, not uploaded files.

    Forcing this one retrieval is the whole feature. Given only a search tool, a 35B on MRCR v2
    declined on 56% of rows, scoring 0.099 when it skipped against 0.461 when it searched; forcing
    the lookup on the evicting turn took tool-only 0.258 to 0.604, and the model then called the
    tool on 0% of rows, so it costs nothing on the common path.

    ``style="tool"`` renders a tool exchange, for the tool loop which already carries a tools array.
    ``style="inline"`` prefixes the latest user message instead, for the plain path: forging
    tool_calls without a tools array breaks strict templates.
    """
    if not thread_id:
        return None
    try:
        from core.rag import conversation_archive
    except Exception:
        return None
    if not conversation_archive.enabled():
        return None

    # The branch's last user turn: the loop conversation may end with an internal nudge.
    query = _last_user_text(branch_messages or conversation) or _last_user_text(conversation)
    if not query:
        return None
    # A nudge ("continue") retrieves nothing, so also query the last real instruction. Not for
    # the model's own search_conversation calls.
    anchor = None
    thin = False
    try:
        from core.inference import instruction_pin
        thin = instruction_pin.is_thin_query(query)
        if thin:
            _behind = branch_messages or conversation
            anchor = instruction_pin.last_substantive_instruction(_behind)
            if not anchor:
                # Use is_thin_query, not a length rule: short prompts can still name something.
                anchor = _last_searchable_text(_behind)
    except Exception:  # noqa: BLE001 -- a query refinement must never break a chat
        anchor = None
        thin = False
    if thin and not anchor:
        # Nothing behind the nudge: skip rather than surface stopword matches.
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
    """Everything checked before pre-retrieval searches: switched on, something to search for,
    somewhere to search, and a store to search it in. Whether a hit then clears the score floor
    is the one part not knowable without running the search. Shared with token counting, which
    cannot run it and so must not decline a turn that stops short of the search here."""
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
    # An unpersisted New Chat has none of the ids.
    if not (rag_scope.get("kb_id") or rag_scope.get("project_id") or thread_id):
        return False, False
    if not _last_user_text(conversation):
        return False, False
    try:
        from storage import rag_db

        # The vec0 native library can be missing from a venv.
        if not rag_db.rag_available():
            return False, False
    except Exception:  # noqa: BLE001
        return False, False
    return bool(enabled), whole_doc_requested


def _thread_document_ids(thread_id) -> set | None:
    """Ids of the thread's indexed attachments; None when the store cannot say."""
    try:
        from core.rag import store
        from storage import rag_db

        conn = rag_db.get_connection()
        try:
            docs = store.list_documents(conn, store.thread_scope(thread_id))
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return None
    return {d["id"] for d in docs if d.get("status") == "completed" and d.get("num_chunks")}


def build_rag_autoinject(conversation: list[dict], rag_scope: dict | None) -> dict | None:
    """Pre-retrieve the latest user turn; if a hit clears the cosine floor return ``{"events": [...],
    "messages": [...]}`` to splice into the loop, else ``None``. Toggle via ``rag_scope.autoinject``
    (else env ``RAG_AUTOINJECT``); floor via ``rag_scope.autoinject_min_score`` (else env
    ``RAG_AUTOINJECT_MIN_SCORE``).

    Also the small-model fallback: models below ~4B often answer from memory instead of calling
    ``search_knowledge_base``, so forcing retrieval here keeps attachments consulted regardless of
    model size.
    """
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
    lean_k = _autoinject_top_k()
    sidebar_k = _opt_int(rag_scope.get("default_top_k"))
    top_k = min(sidebar_k, lean_k) if sidebar_k is not None and sidebar_k > 0 else lean_k
    budget: int | None = None
    # Only trust a GGUF actually serving this same window.
    ctx_tokens = (
        _opt_int(rag_scope.get("context_length") or rag_scope.get("max_context_tokens")) or 0
    )

    def _fits(candidate_text, max_tokens) -> bool:
        if max_tokens is None:
            return True
        if max_tokens <= 0:
            return False
        # Priced by the GGUF when possible, doubled otherwise, so dense ASCII is not undercharged.
        return _text_token_cost(candidate_text, ctx_tokens) <= max_tokens

    def _trim(
        hit_text,
        hit_sources,
        max_tokens,
        keep_first = 1,
    ):
        """Drop passages from the tail until the rendered block fits, else None. Re-renders only
        when something is dropped. None when not even the first ``keep_first`` passages fit: the
        block joins the current turn, which the window may not evict, so it fails the request
        rather than degrading it. ``keep_first`` is the floor of the tail: one for ranked
        retrieval, but a whole document must never be eaten into."""
        floor = max(1, keep_first)
        kept, rendered = list(hit_sources), hit_text
        while len(kept) > floor and not _fits(rendered, max_tokens):
            kept = kept[:-1]
            rendered = render_sources(kept)
        return (rendered, kept) if _fits(rendered, max_tokens) else None

    # Thread attachments under budget go in whole. KB selection is exclusive; project sources are
    # still top-K. Oversized thread docs fall through to top-K.
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
                    # The document stays whole; only the project tail is trimmed.
                    merged = sources + proj[1]
                    trimmed = _trim(render_sources(merged), merged, budget, keep_first = len(sources))
                    if trimmed is not None:
                        text, sources = trimmed
            logger.info("RAG auto-inject: whole-document context (%d chunk(s))", len(sources))

    def retrieve(*, max_tokens = None, **scope):
        found = search_for_autoinject(query = query, top_k = top_k, **scope)
        return _trim(found[0], found[1], max_tokens) if found else None

    thread_docs = _thread_document_ids(thread_id) if whole_doc_requested and text is None else set()

    def retrieve_thread_unfloored(*, max_tokens = None):
        # Lexical-only misses generic requests ("summarize this"), so retry with the dense leg.
        if thread_docs is not None and not thread_docs:
            return None
        scope_kwargs = _scope_retrieval_kwargs(rag_scope)
        found = retrieve(
            max_tokens = max_tokens, scope_thread_id = thread_id, min_dense_score = None, **scope_kwargs
        )
        if not found and scope_kwargs["mode"] == "lexical":
            found = retrieve(
                max_tokens = max_tokens,
                scope_thread_id = thread_id,
                min_dense_score = None,
                mode = "hybrid",
            )
        return found

    # An oversized attachment is mandatory grounding: search it alone without the relevance floor,
    # then add project context if it fits.
    if text is None and (enabled or whole_doc_requested):
        try:
            if whole_doc_requested and not enabled:
                found = retrieve_thread_unfloored(max_tokens = budget)
                project_id = rag_scope.get("project_id")
                if found and project_id:
                    # Isolated so an unavailable project index does not discard thread grounding.
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
                # Do not let project hits crowd out the attachment.
                grounded = (
                    bool(found)
                    and thread_docs is not None
                    and any(s.get("documentId") in thread_docs for s in found[1])
                )
                if (
                    whole_doc_requested
                    and (not found or rag_scope.get("project_id"))
                    and not grounded
                ):
                    thread_found = retrieve_thread_unfloored()
                    if thread_found and found:
                        cited = thread_docs or {s.get("documentId") for s in thread_found[1]}
                        if not any(s.get("documentId") in cited for s in found[1]):
                            n_proj = min(len(found[1]), top_k // 2)
                            merged = thread_found[1][: top_k - n_proj] + found[1][:n_proj]
                            found = (render_sources(merged), merged)
                    elif thread_found:
                        found = thread_found
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


_MAX_PAGE_CHARS = 16000  # fetched page cap after HTML-to-Markdown conversion

# one page may use 35% of the window, leaving room for prompts, context, calls, and answers.
_PAGE_CONTEXT_SHARE = 0.35
# unmeasurable token budgets are halved because dense ASCII can cost twice the English estimate.
_UNMEASURED_ROOM_MARGIN = 0.5

_MIN_PAGE_CHARS = 2000
# a percent escape is one non-ASCII byte in ASCII and tokenizes like one.
_HEX_PAIR_RE = re.compile(r"[0-9A-Fa-f]{2}")
# raw cap exceeds _MAX_PAGE_CHARS because conversion strips large SSR <head> sections.
_MAX_FETCH_BYTES = 512 * 1024
# news pages can inline about 2.5 MB in <head>, so reserve _MAX_FETCH_BYTES beyond its end.
_MAX_HTML_FETCH_BYTES = 8 * 1024 * 1024
# keep % safe to avoid encoding existing escapes as %25.
_IRI_PATH_SAFE = "/%:@!$&'()*+,;="
_IRI_QUERY_SAFE = "/%:@!$&'()*+,;=?"
# PDF cross-reference data at EOF requires the whole body.
_MAX_PDF_FETCH_BYTES = 10 * 1024 * 1024
_MAX_WEB_PDF_PAGES = 50
# binary threshold excludes whitespace and ESC; allow 16 glitches or 12.5%, whichever is larger.
_BINARY_CHAR_RE = re.compile("[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1a\\x1c-\\x1f\\x7f-\\x9f\\ufffd]")
_MIN_BINARY_CHARS = 16
_BINARY_CHAR_DIVISOR = 8
# signatures catch mislabeled binaries that pass text heuristics.
_PDF_MAGIC = b"%PDF-"
_BINARY_MAGIC = (
    _PDF_MAGIC,
    b"PK\x03\x04",
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"\x1f\x8b",
    b"BZh",
    b"\xfd7zXZ\x00",
    b"\x28\xb5\x2f\xfd",
)

# Check UTF-32 first because its little-endian BOM starts with the UTF-16 BOM.
_UNICODE_BOM_CODECS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
    (codecs.BOM_UTF8, "utf-8-sig"),
)

_MIN_SINGLE_BYTE_ASCII_RATIO = 3 / 4
_ASCII_TEXT_BYTES = frozenset((*range(0x20, 0x7F), 0x09, 0x0A, 0x0D, 0x1B))

_META_CHARSET_SCAN_BYTES = 2048
_META_REFRESH_SCAN_BYTES = 4096
_META_REFRESH_MAX_DELAY = 5
_META_REFRESH_CONTENT_RE = re.compile(
    r"\s*(\d+|(?=\.))(?:\.[\d.]*)?(?:(?=[\s;,])\s*[;,]?\s*(?:url\s*=\s*)?(.*))?\Z",
    re.ASCII | re.IGNORECASE | re.DOTALL,
)
# Browsers leave an unterminated named reference in an attribute alone, so "&section=" stays literal instead of "§ion=".
_ATTR_CHAR_REF_RE = re.compile(r"&(?:#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[A-Za-z][A-Za-z0-9]*;)")
_META_REFRESH_INERT_TAGS = frozenset(
    (
        b"iframe",
        b"noembed",
        b"noframes",
        b"noscript",
        b"plaintext",
        b"script",
        b"style",
        b"template",
        b"textarea",
        b"title",
        b"xmp",
    )
)
# Comments and whole tags, so markup inside is never read as <meta>.
_HTML_TAG_RE = re.compile(
    rb"<!--.*?(?:-->|\Z)|<([a-z][^\s/>]*)((?:[\s/](?:[^>\"']|\"[^\"]*\"|'[^']*')*)?)>|<[!/?][^>]*>|<[a-z!/?].*",
    re.IGNORECASE | re.DOTALL,
)
_META_ATTR_RE = re.compile(rb"([^\s\"'/=>]+)(?:\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]*)))?")
_META_CONTENT_CHARSET_RE = re.compile(rb"charset\s*=\s*[\"']?\s*([^\s\"';]+)", re.IGNORECASE)
_XML_ENCODING_RE = re.compile(rb"\s*<\?xml\b[^>]*?encoding\s*=\s*[\"']([\w.:-]+)", re.IGNORECASE)
# UTF-16 or x-user-defined meta declarations mean UTF-8 or windows-1252 to browsers.
_WHATWG_CHARSET_LABELS = {
    "utf-8": (
        "unicode-1-1-utf-8 unicode11utf8 unicode20utf8 utf-8 utf8 x-unicode20utf8 unicodefffe "
        "utf-16be csunicode iso-10646-ucs-2 ucs-2 unicode unicodefeff utf-16 utf-16le"
    ),
    "cp866": "866 cp866 csibm866 ibm866",
    "iso8859-2": (
        "csisolatin2 iso-8859-2 iso-ir-101 iso8859-2 iso88592 iso_8859-2 iso_8859-2:1987 l2 "
        "latin2"
    ),
    "iso8859-3": (
        "csisolatin3 iso-8859-3 iso-ir-109 iso8859-3 iso88593 iso_8859-3 iso_8859-3:1988 l3 "
        "latin3"
    ),
    "iso8859-4": (
        "csisolatin4 iso-8859-4 iso-ir-110 iso8859-4 iso88594 iso_8859-4 iso_8859-4:1988 l4 "
        "latin4"
    ),
    "iso8859-5": (
        "csisolatincyrillic cyrillic iso-8859-5 iso-ir-144 iso8859-5 iso88595 iso_8859-5 "
        "iso_8859-5:1988"
    ),
    "iso8859-6": (
        "arabic asmo-708 csiso88596e csiso88596i csisolatinarabic ecma-114 iso-8859-6 "
        "iso-8859-6-e iso-8859-6-i iso-ir-127 iso8859-6 iso88596 iso_8859-6 iso_8859-6:1987"
    ),
    "iso8859-7": (
        "csisolatingreek ecma-118 elot_928 greek greek8 iso-8859-7 iso-ir-126 iso8859-7 iso88597 "
        "iso_8859-7 iso_8859-7:1987 sun_eu_greek"
    ),
    "iso8859-8": (
        "csiso88598e csisolatinhebrew hebrew iso-8859-8 iso-8859-8-e iso-ir-138 iso8859-8 "
        "iso88598 iso_8859-8 iso_8859-8:1988 visual csiso88598i iso-8859-8-i logical"
    ),
    "iso8859-10": "csisolatin6 iso-8859-10 iso-ir-157 iso8859-10 iso885910 l6 latin6",
    "iso8859-13": "iso-8859-13 iso8859-13 iso885913",
    "iso8859-14": "iso-8859-14 iso8859-14 iso885914",
    "iso8859-15": "csisolatin9 iso-8859-15 iso8859-15 iso885915 iso_8859-15 l9",
    "iso8859-16": "iso-8859-16",
    "koi8-r": "cskoi8r koi koi8 koi8-r koi8_r",
    "koi8-u": "koi8-ru koi8-u",
    "mac-roman": "csmacintosh mac macintosh x-mac-roman",
    "cp874": "dos-874 iso-8859-11 iso8859-11 iso885911 tis-620 windows-874",
    "cp1250": "cp1250 windows-1250 x-cp1250",
    "cp1251": "cp1251 windows-1251 x-cp1251",
    "cp1252": (
        "ansi_x3.4-1968 ascii cp1252 cp819 csisolatin1 ibm819 iso-8859-1 iso-ir-100 iso8859-1 "
        "iso88591 iso_8859-1 iso_8859-1:1987 l1 latin1 us-ascii windows-1252 x-cp1252 "
        "x-user-defined"
    ),
    "cp1253": "cp1253 windows-1253 x-cp1253",
    "cp1254": (
        "cp1254 csisolatin5 iso-8859-9 iso-ir-148 iso8859-9 iso88599 iso_8859-9 iso_8859-9:1989 "
        "l5 latin5 windows-1254 x-cp1254"
    ),
    "cp1255": "cp1255 windows-1255 x-cp1255",
    "cp1256": "cp1256 windows-1256 x-cp1256",
    "cp1257": "cp1257 windows-1257 x-cp1257",
    "cp1258": "cp1258 windows-1258 x-cp1258",
    "mac-cyrillic": "x-mac-cyrillic x-mac-ukrainian",
    "gb18030": (
        "chinese csgb2312 csiso58gb231280 gb2312 gb_2312 gb_2312-80 gbk iso-ir-58 x-gbk gb18030"
    ),
    "big5hkscs": "big5 big5-hkscs cn-big5 csbig5 x-x-big5",
    "euc_jp": "cseucpkdfmtjapanese euc-jp x-euc-jp",
    "iso2022_jp": "csiso2022jp iso-2022-jp",
    "cp932": "csshiftjis ms932 ms_kanji shift-jis shift_jis sjis windows-31j x-sjis",
    "cp949": (
        "cseuckr csksc56011987 euc-kr iso-ir-149 korean ks_c_5601-1987 ks_c_5601-1989 ksc5601 "
        "ksc_5601 windows-949"
    ),
}
_WHATWG_CHARSET_CODECS = {
    label: codec for codec, labels in _WHATWG_CHARSET_LABELS.items() for label in labels.split()
}


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


def _whatwg_codec(label: bytes) -> str | None:
    return _WHATWG_CHARSET_CODECS.get(label.strip(b"\t\n\f\r ").decode("latin-1").lower())


def _sniff_meta_charset(head: bytes, content_type: str) -> str | None:
    # Browsers prescan <meta> only in HTML and read only the prolog in XML.
    prolog = _XML_ENCODING_RE.match(head)
    if content_type:
        is_html = content_type == "text/html"
        is_xml = content_type in ("text/xml", "application/xml") or content_type.endswith("+xml")
    else:
        is_xml = prolog is not None
        is_html = not is_xml and _looks_like_html(head.decode("latin-1"))
    if is_html:
        for tag in _HTML_TAG_RE.finditer(head):
            if (tag.group(1) or b"").lower() != b"meta":
                continue
            attrs = {}
            for name, *values in _META_ATTR_RE.findall(tag.group(2)):
                attrs.setdefault(name.lower(), b"".join(values))
            label = attrs.get(b"charset")
            if label is None and attrs.get(b"http-equiv", b"").strip().lower() == b"content-type":
                match = _META_CONTENT_CHARSET_RE.search(attrs.get(b"content", b""))
                label = match and match.group(1)
            codec = label and _whatwg_codec(label)
            if codec:
                return codec
    elif not is_xml:
        return None
    return prolog and _whatwg_codec(prolog.group(1))


def _meta_refresh_target(body: bytes, page_url: str) -> str | None:
    from html import unescape
    from urllib.parse import urldefrag, urljoin, urlparse

    def attr_text(value: bytes) -> str:
        return _ATTR_CHAR_REF_RE.sub(
            lambda ref: unescape(ref.group(0)), value.decode("utf-8", "replace")
        )

    base_url, seen_base, inert = page_url, False, None
    for tag in _HTML_TAG_RE.finditer(body[:_META_REFRESH_SCAN_BYTES]):
        name = (tag.group(1) or b"").lower()
        if inert is not None:
            end = tag.group(0)[: len(inert) + 3].lower()
            if inert != b"plaintext" and end[:-1] == b"</" + inert and end[-1:] in b"\t\n\f\r />":
                inert = None
        elif name in _META_REFRESH_INERT_TAGS:
            inert = name
        elif name in (b"base", b"meta"):
            attrs = {}
            for attr, *values in _META_ATTR_RE.findall(tag.group(2)):
                attrs.setdefault(attr.lower(), b"".join(values))
            if name == b"base":
                if b"href" in attrs and not seen_base:
                    seen_base = True
                    try:
                        href = urljoin(page_url, attr_text(attrs[b"href"]).strip())
                        if urlparse(href).scheme in ("http", "https"):
                            base_url = href
                    except ValueError:
                        pass
                continue
            if attrs.get(b"http-equiv", b"").strip().lower() != b"refresh":
                continue
            match = _META_REFRESH_CONTENT_RE.match(attr_text(attrs.get(b"content", b"")))
            if match is None:
                continue
            location = (match.group(2) or "").strip()
            if location[:1] in ("'", '"'):
                location = location[1:].split(location[0], 1)[0].strip()
            if not location or int(match.group(1) or 0) > _META_REFRESH_MAX_DELAY:
                return None
            try:
                target = urljoin(base_url, location)
                scheme = urlparse(target).scheme
            except ValueError:
                return None
            if scheme not in ("http", "https") or urldefrag(target)[0] == urldefrag(page_url)[0]:
                return None
            return target
    return None


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


_PINNED_DIAL_TIMEOUT = 1.0
_PINNED_PROBE_BUDGET = 0.25


def _pinned_create_connection(addresses):
    """``socket.create_connection`` that walks *addresses* instead of resolving.

    Pinning to one validated address dropped the walk every other client gets. Walking
    here rather than around ``opener.open`` keeps the caller's timeout covering one
    whole response.
    """
    import socket

    def create(
        address,
        timeout = socket._GLOBAL_DEFAULT_TIMEOUT,
        source_address = None,
    ):
        if timeout is socket._GLOBAL_DEFAULT_TIMEOUT:
            timeout = socket.getdefaulttimeout()
        if address[0] not in addresses:
            return socket.create_connection(address, timeout, source_address)

        port = address[1]
        expiry = None if timeout is None else time.monotonic() + timeout
        probe_timeout = (
            None
            if timeout is None
            else min(
                timeout * _PINNED_PROBE_BUDGET / len(addresses),
                _PINNED_DIAL_TIMEOUT,
            )
        )
        error = None
        # A brief look at every address first, so one silent address cannot block the rest.
        passes = (True, False) if len(addresses) > 1 and timeout is not None else (False,)
        for probe in passes:
            for index, ip in enumerate(addresses):
                if timeout is None:
                    dial_timeout = None
                elif probe:
                    dial_timeout = probe_timeout
                else:
                    dial_timeout = max(
                        (expiry - time.monotonic()) / (len(addresses) - index),
                        0,
                    )
                try:
                    sock = socket.create_connection((ip, port), dial_timeout, source_address)
                except OSError as exc:
                    error = exc
                    continue
                sock.settimeout(timeout)
                return sock
        raise error

    return create


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection to a pinned IP, using a different hostname for SNI and
    cert verification.

    SSRF IP-pinning rewrites URLs to raw IPs; a normal HTTPSConnection would then
    send no SNI and verify the cert against the IP (both fail). This splits the
    concerns: TCP connects to a validated IP, TLS uses ``sni_hostname``.
    """

    def __init__(
        self,
        host: str,
        *,
        sni_hostname: str,
        addresses = (),
        **kwargs,
    ):
        super().__init__(host, **kwargs)
        self._sni_hostname = sni_hostname
        if addresses:
            self._create_connection = _pinned_create_connection(tuple(addresses))

    def connect(self):
        http.client.HTTPConnection.connect(self)
        self.sock = self._context.wrap_socket(
            self.sock,
            server_hostname = self._sni_hostname,
        )


class _SNIHTTPSHandler(urllib.request.HTTPSHandler):
    """HTTPS handler sending the correct SNI hostname during TLS handshake.

    SSRF IP-pinning breaks SNI and cert verification; this returns a
    ``_PinnedHTTPSConnection`` that connects to the pinned IP but verifies TLS
    against the original hostname. *addresses* are every validated address, pinned
    one first, for ``_pinned_create_connection`` to walk.
    """

    def __init__(
        self,
        hostname: str,
        addresses = (),
    ):
        super().__init__(context = _tls_ctx)
        self._sni_hostname = hostname
        self._addresses = tuple(addresses)

    def https_open(self, req):
        return self.do_open(self._sni_connection, req)

    def _sni_connection(self, host, **kwargs):
        kwargs["context"] = _tls_ctx
        return _PinnedHTTPSConnection(
            host,
            sni_hostname = self._sni_hostname,
            addresses = self._addresses,
            **kwargs,
        )


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, addresses = ()):
        super().__init__()
        self._addresses = tuple(addresses)

    def http_open(self, req):
        return self.do_open(self._pinned_connection, req)

    def _pinned_connection(self, host, **kwargs):
        conn = http.client.HTTPConnection(host, **kwargs)
        if self._addresses:
            conn._create_connection = _pinned_create_connection(self._addresses)
        return conn


def _explicit_proxy_applies(scheme: str, host: str) -> bool:
    """Whether urllib routes a *scheme* request for *host* through a proxy.

    Only a proxied fetch may keep the hostname in the request URL: the proxy resolves it, so this
    host never looks it up again. A direct one would, which is the DNS-rebinding window, so it stays
    pinned to the validated IP.

    *host* must be the ``host[:port]`` form ``Request.host`` carries, since that is what
    ``ProxyHandler`` passes to ``proxy_bypass``; probing the bare hostname instead would disagree
    with it on a port-qualified NO_PROXY entry.
    """
    from urllib.request import getproxies, proxy_bypass

    # ProxyHandler lowercases keys and the Windows registry can return "HTTPS=...".
    if scheme not in {key.lower() for key in getproxies()}:
        return False
    try:
        return not proxy_bypass(host)
    except (OSError, ValueError):
        return False


def _validate_and_resolve_host(hostname: str, port: int) -> tuple[bool, str, list[str]]:
    """Resolve *hostname*, reject non-public IPs, return the pinned IP strings.

    Returns ``(ok, reason_or_empty, resolved_ips)`` in resolver order. The caller pins
    to these (with a ``Host`` header) rather than resolving again, which is the DNS
    rebinding window, and walks them as ``socket.create_connection`` would.
    """
    import ipaddress
    import socket

    try:
        infos = socket.getaddrinfo(hostname, port, type = socket.SOCK_STREAM)
    # IDNA encoding rejects a hostname with UnicodeError, not OSError.
    except (OSError, UnicodeError) as e:
        return False, f"Failed to resolve host: {e}", []

    if not infos:
        return False, f"Failed to resolve host: no addresses for {hostname!r}", []

    resolved = []
    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        # `not ip.is_global` decides; the explicit predicates only label the error.
        if (
            not ip.is_global
            or ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False, f"Blocked: refusing to fetch non-public address {ip}.", []
        if sockaddr[0] not in resolved:
            resolved.append(sockaddr[0])

    return True, "", resolved


# Binary application subtypes are rejected; others are sniffed so text like SQL stays usable.
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
    """README API URL for a ``github.com/{owner}/{repo}`` page, else None. A repo root page rendered
    as HTML is mostly UI chrome (nav, file table, stats); the ``/readme`` API returns the raw
    README markdown unauthenticated, which is what the model actually wants to read."""
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


# One wall-clock deadline (plus cancel_event) bounds a multi-step fetch; socket timeouts
# bound only one step.
def _fetch_budget_exceeded(deadline, cancel_event):
    """User-facing error string when the fetch must stop early, else None."""
    if cancel_event is not None and cancel_event.is_set():
        return "Failed to fetch URL: cancelled."
    if deadline is not None and time.monotonic() >= deadline:
        return "Failed to fetch URL: timed out."
    return None


def _fetch_hop_timeout(timeout, deadline):
    """Per-operation socket timeout: the lesser of the caller's per-op timeout and the time left on
    the deadline, so one slow hop cannot overrun the whole budget. Callers check
    ``_fetch_budget_exceeded`` first, so remaining time is positive here; the tiny floor only
    guards a race."""
    if deadline is None:
        return timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        remaining = 0.001
    return remaining if timeout is None else min(timeout, remaining)


def _resolve_with_budget(hostname, port, deadline, cancel_event):
    """``_validate_and_resolve_host`` bounded by the overall fetch budget. ``getaddrinfo`` is
    blocking with no deadline of its own, so a slow resolver (or a request cancelled before
    dispatch) could run past the budget. Resolve on a daemon thread and poll the budget so the
    fetch aborts on time; the abandoned lookup is discarded. With no deadline and no cancel_event
    this is a plain synchronous call, so opt-out callers keep the old behavior and cost."""
    budget_error = _fetch_budget_exceeded(deadline, cancel_event)
    if budget_error is not None:
        return False, budget_error, []
    if deadline is None and cancel_event is None:
        return _validate_and_resolve_host(hostname, port)

    result: "queue.Queue" = queue.Queue(maxsize = 1)

    def _resolve():
        try:
            result.put(_validate_and_resolve_host(hostname, port))
        except Exception as exc:  # prevent the resolver thread from failing silently
            result.put((False, f"Failed to resolve host: {exc}", []))

    threading.Thread(target = _resolve, name = "web-fetch-dns", daemon = True).start()
    while True:
        budget_error = _fetch_budget_exceeded(deadline, cancel_event)
        if budget_error is not None:
            return False, budget_error, []
        try:
            return result.get(timeout = 0.05)
        except queue.Empty:
            continue


class _HTMLBodyLocator(HTMLParser):
    """locate the explicit or implied document body without matching markup inside head content."""

    _PENDING_LIMIT = 65536
    _CDATA_TAIL_BYTES = 64
    _HEAD_ELEMENTS = frozenset(
        {
            "base",
            "basefont",
            "bgsound",
            "link",
            "meta",
            "noframes",
            "noscript",
            "script",
            "style",
            "template",
            "title",
        }
    )
    _HEAD_TEXT_ELEMENTS = frozenset({"noframes", "noscript", "script", "style", "title"})

    def __init__(self, charset = None):
        super().__init__(convert_charrefs = False)
        self.body_at = None
        self._absolute_offset = 0
        self._head_text_depth = 0
        self._template_depth = 0
        self._prefix = b""
        self._decoder = None
        self._codec = "latin-1"
        self._deferred = []
        self._deferred_chars = 0
        try:
            codec = codecs.lookup(charset).name if charset else None
        except (LookupError, ValueError):
            codec = None
        if codec in ("utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-32-le", "utf-32-be"):
            self._codec = codec if codec.endswith(("-le", "-be")) else codec + "-le"

    def feed_bytes(self, data):
        if self._decoder is None:
            self._prefix += data
            if len(self._prefix) < 4:
                return
            data, self._prefix = self._prefix, b""
            for bom, codec in (
                (codecs.BOM_UTF32_LE, "utf-32-le"),
                (codecs.BOM_UTF32_BE, "utf-32-be"),
                (codecs.BOM_UTF16_LE, "utf-16-le"),
                (codecs.BOM_UTF16_BE, "utf-16-be"),
            ):
                if data.startswith(bom):
                    self._codec = codec
                    self._absolute_offset = len(bom)
                    data = data[len(bom) :]
                    break
            self._decoder = codecs.getincrementaldecoder(self._codec)(errors = "replace")
        decoded = self._decoder.decode(data)
        if len(self.rawdata) > self._PENDING_LIMIT and not self.cdata_elem:
            self._deferred.append(decoded)
            self._deferred_chars += len(decoded)
            if self._deferred_chars < len(self.rawdata):
                return
            decoded = "".join(self._deferred)
            self._deferred.clear()
            self._deferred_chars = 0
        self.feed(decoded)
        if self.body_at is not None or len(self.rawdata) <= self._PENDING_LIMIT:
            return
        if self.cdata_elem:
            discard = len(self.rawdata) - self._CDATA_TAIL_BYTES
            self.updatepos(0, discard)
            self.rawdata = self.rawdata[discard:]

    def updatepos(self, i, j):
        if j > i:
            self._absolute_offset += (
                j - i
                if self._codec == "latin-1"
                else len(self.rawdata[i:j].encode(self._codec, errors = "replace"))
            )
        return super().updatepos(i, j)

    def _offset(self):
        return self._absolute_offset

    def _mark_body(self):
        if self.body_at is None:
            self.body_at = self._offset()

    def handle_starttag(self, tag, attrs):
        if self.body_at is not None:
            return
        if self._template_depth:
            if tag == "template":
                self._template_depth += 1
            return
        if tag == "template":
            self._template_depth = 1
            return
        if self._head_text_depth:
            if tag in self._HEAD_TEXT_ELEMENTS:
                self._head_text_depth += 1
            return
        if tag in ("html", "head"):
            return
        if tag == "body":
            self._mark_body()
            return
        if tag in self._HEAD_ELEMENTS:
            if tag in self._HEAD_TEXT_ELEMENTS:
                self._head_text_depth = 1
            return
        self._mark_body()

    def handle_startendtag(self, tag, attrs):
        if (
            self.body_at is None
            and not self._template_depth
            and not self._head_text_depth
            and tag not in self._HEAD_ELEMENTS
            and tag not in ("html", "head")
        ):
            self._mark_body()

    def handle_endtag(self, tag):
        if self.body_at is not None:
            return
        if self._template_depth:
            if tag == "template":
                self._template_depth -= 1
            return
        if self._head_text_depth:
            if tag in self._HEAD_TEXT_ELEMENTS:
                self._head_text_depth -= 1
            return
        if tag == "head":
            self._mark_body()

    def handle_data(self, data):
        if self._offset() == 0 and data.startswith(codecs.BOM_UTF8.decode("latin-1")):
            data = data[len(codecs.BOM_UTF8) :]
        if (
            self.body_at is None
            and not self._template_depth
            and not self._head_text_depth
            and data.strip()
        ):
            self._mark_body()

    def handle_entityref(self, name):
        from html import unescape
        self.handle_data(unescape("&" + name + ";"))

    def handle_charref(self, name):
        self.handle_entityref("#" + name)


def _read_capped_body(
    resp,
    max_bytes,
    timeout,
    deadline,
    cancel_event,
    body_window = None,
    charset = None,
):
    """read at most ``max_bytes``, and ``body_window`` past the end of ``<head>``; returns ``(error, body)``."""
    # HTTPError exposes the socket for deadline updates; chunk checks bound test doubles without one
    fp = getattr(resp, "fp", None)
    sock = getattr(getattr(getattr(fp, "fp", fp), "raw", None), "_sock", None)
    # read1 avoids buffered read(n) waiting for n bytes past the budget
    read = getattr(resp, "read1", None) or resp.read
    chunks = []
    remaining = max_bytes
    body_at = None
    body_locator = _HTMLBodyLocator(charset) if body_window is not None else None
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
        chunk = read(min(65536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
        if body_locator is not None and body_at is None:
            got = max_bytes - remaining
            try:
                body_locator.feed_bytes(chunk)
            except Exception:
                body_locator = None
            if body_locator is not None and body_locator.body_at is not None:
                body_at = body_locator.body_at
                remaining = max(0, min(remaining, body_at + body_window - got))
    budget_error = _fetch_budget_exceeded(deadline, cancel_event)
    if budget_error is not None:
        try:
            resp.close()
        except Exception:
            pass
        return budget_error, b""
    return None, b"".join(chunks)


_DOTTED_HOST_RE = re.compile(r"[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+")
# ASCII-only: str.isdigit() accepts digits int() rejects; five digits bounds integer conversion
_PORT_RE = re.compile(r"[0-9]{1,5}")


def _normalize_url_scheme(url: str) -> str:
    """Prepend ``https://`` to bare hosts (``google.com``, ``example.com:8443``).

    ``urlparse`` reads the host of a ``host:port`` input as the scheme, so those are recognised by a
    dotted host-like scheme with an empty netloc. Rewrites a dotted host with an optional in-range
    port, and the ``//host`` form. Real schemes (``file:``, ``javascript:``, including ``file:80``),
    root-relative paths (``/login``) and bad ports are returned untouched so the caller rejects
    them. A dotted scheme is indistinguishable from ``host:port``, so ``com.acme.app:443/cb`` is
    rewritten too; an empty port is kept as-is.

    The host is matched against the raw authority, never against what ``urlparse`` returned, because
    urlsplit strips tabs/newlines (3.10) and leading C0/space (3.12). Anything it would strip fails
    the match, so the decision and the rewritten string cannot disagree across versions.
    """
    from urllib.parse import urlparse

    url = url.strip()
    try:
        parsed = urlparse(url)
    except ValueError:
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


def _pinned_netloc(ip: str, port: int | None) -> str:
    host = f"[{ip}]" if ":" in ip else ip
    return f"{host}:{port}" if port else host


def _redirect_hop(url: str, website_policy, deadline, cancel_event) -> tuple[str | None, str, list]:
    from urllib.parse import urlparse
    from .web_access_policy import check_url_access

    allowed, reason, host = check_url_access(url, website_policy)
    if not allowed:
        return reason, "", []
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    ok, reason, pinned_ips = _resolve_with_budget(host, port, deadline, cancel_event)
    if not ok:
        return reason, "", []
    return None, host, pinned_ips


def _is_bot_check(status: int, headers) -> bool:
    """Whether a refusal came from a bot check (Cloudflare, DataDome, Akamai), not the site."""
    if headers is None:
        return False
    if (headers.get("cf-mitigated") or "").lower() == "challenge":
        return True
    if headers.get("x-datadome") or headers.get("x-dd-b"):
        return True
    server = (headers.get("Server") or "").lower()
    # Rate limits and outages behind these CDNs carry the same Server header.
    return status == 403 and ("cloudflare" in server or "akamaighost" in server)


def _fetch_url_raw(
    url: str,
    timeout: int = 30,
    extra_headers: dict | None = None,
    deadline: float | None = None,
    cancel_event = None,
    website_policy: dict | None = None,
    raw_bytes_max: int | None = None,
    post_data: bytes | None = None,
    meta_out: dict | None = None,
    host_headers = None,
    error_page: bool = False,
) -> tuple[str | None, "str | bytes", str]:
    """fetch with SSRF protection; binary reads stay capped, HTML error pages require binary mode, per-hop headers do not cross redirects, and deadlines cover redirects and body reads."""
    from urllib.parse import urlparse
    from .web_access_policy import check_url_access

    # normalize before the policy gate because a bare host would otherwise fail its http(s) scheme check.
    url = _normalize_url_scheme(url)
    allowed, reason, canonical_host = check_url_access(url, website_policy)
    if not allowed:
        return reason, "", ""

    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    ok, reason, pinned_ips = _resolve_with_budget(
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
        pending_post = post_data
        http_error = None

        for _hop in range(5):
            budget_error = _fetch_budget_exceeded(deadline, cancel_event)
            if budget_error is not None:
                return budget_error, "", ""
            cp = urlparse(current_url)
            cp = cp._replace(
                path = quote(cp.path, safe = _IRI_PATH_SAFE),
                params = quote(cp.params, safe = _IRI_PATH_SAFE),
                query = quote(cp.query, safe = _IRI_QUERY_SAFE),
            )
            validated_netloc = f"[{current_host}]" if ":" in current_host else current_host
            if cp.port:
                validated_netloc = f"{validated_netloc}:{cp.port}"
            # Decide routing once, on the netloc: a pinned request carries an IP no NO_PROXY entry matches.
            proxied = _explicit_proxy_applies(cp.scheme, validated_netloc)
            if os.environ.get(_DISABLE_DNS_PINNING_ENV) == "1" and proxied:
                # Proxies need the hostname in CONNECT and resolve it themselves.
                request_url = urlunparse(cp._replace(netloc = validated_netloc))
            else:
                # Pin the first validated IP against DNS rebinding.
                request_url = urlunparse(cp._replace(netloc = _pinned_netloc(pinned_ips[0], cp.port)))

            walk = () if proxied else pinned_ips
            handlers = [
                _NoRedirect,
                _SNIHTTPSHandler(current_host, walk),
                _PinnedHTTPHandler(walk),
            ]
            if not proxied:
                handlers.append(urllib.request.ProxyHandler({}))
            opener = urllib.request.build_opener(*handlers)

            headers = {
                "User-Agent": ua,
                "Host": validated_netloc,
            }
            if extra_headers:
                headers.update(extra_headers)
            if host_headers is not None:
                headers.update(host_headers(current_host))
            if pending_post is not None:
                headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
            req = urllib.request.Request(request_url, headers = headers, data = pending_post)
            try:
                # cap the socket timeout at the remaining deadline so one slow hop cannot outlast the fetch budget.
                resp = opener.open(req, timeout = _fetch_hop_timeout(timeout, deadline))
            except _HTTPError as e:
                if e.code not in (301, 302, 303, 307, 308):
                    if meta_out is not None:
                        meta_out["bot_check"] = _is_bot_check(e.code, e.headers)
                    http_error = f"Failed to fetch URL: HTTP {e.code} {getattr(e, 'reason', '')}"
                    declared = e.headers.get("Content-Type") and e.headers.get_content_type()
                    if (
                        not error_page
                        or raw_bytes_max is None
                        or declared not in (None, "", "text/html", "application/xhtml+xml")
                    ):
                        return http_error, "", ""
                    resp = e
                else:
                    location = e.headers.get("Location")
                    if not location:
                        return "Failed to fetch URL: redirect missing Location header.", "", ""
                    current_url = urljoin(current_url, location)
                    # 307/308 preserve POST; other redirects switch to GET.
                    if e.code not in (307, 308):
                        pending_post = None
                    hop_error, current_host, pinned_ips = _redirect_hop(
                        current_url,
                        website_policy,
                        deadline,
                        cancel_event,
                    )
                    if hop_error is not None:
                        return hop_error, "", ""
                    continue

            # get_content_type() maps absent headers to text/plain per RFC 2045; "" marks absence.
            if resp.headers.get("Content-Type") is None:
                content_type = ""
            else:
                content_type = (resp.headers.get_content_type() or "").lower()

            # chunked reads recheck the fetch budget between chunks.
            declared_pdf = raw_bytes_max is None and content_type == "application/pdf"
            declared_html = raw_bytes_max is None and content_type in (
                "text/html",
                "application/xhtml+xml",
            )
            if raw_bytes_max is not None:
                read_limit = raw_bytes_max + 1
            elif declared_pdf:
                read_limit = _MAX_PDF_FETCH_BYTES + 1
            elif declared_html:
                read_limit = _MAX_HTML_FETCH_BYTES
            else:
                read_limit = max_bytes
            body_error, raw_bytes = _read_capped_body(
                resp,
                read_limit,
                timeout,
                deadline,
                cancel_event,
                body_window = max_bytes if declared_html else None,
                charset = resp.headers.get_content_charset() if declared_html else None,
            )
            if body_error is not None:
                return body_error, "", ""

            # missing or wrong PDF MIME types require a bounded tail read to reach the EOF xref.
            if raw_bytes_max is not None:
                if len(raw_bytes) > raw_bytes_max:
                    return f"(content exceeds the {raw_bytes_max} byte limit)", "", content_type
                if meta_out is not None:
                    meta_out["url"] = current_url
                    meta_out["charset"] = resp.headers.get_content_charset()
                    meta_out["filename"] = resp.headers.get_filename()
                    meta_out["allow_origin"] = resp.headers.get("Access-Control-Allow-Origin")
                    meta_out["cache_control"] = resp.headers.get("Cache-Control")
                    meta_out["age"] = resp.headers.get("Age")
                return http_error, raw_bytes, content_type
            if not declared_pdf and _has_pdf_magic(raw_bytes):
                tail_error, tail = _read_capped_body(
                    resp,
                    _MAX_PDF_FETCH_BYTES - len(raw_bytes) + 1,
                    timeout,
                    deadline,
                    cancel_event,
                )
                if tail_error is not None:
                    return tail_error, "", ""
                raw_bytes += tail
            refresh_url = content_type == "text/html" and _meta_refresh_target(
                raw_bytes, current_url
            )
            if not refresh_url:
                break
            current_url = refresh_url
            # a refresh starts a new GET like a browser.
            pending_post = None
            hop_error, current_host, pinned_ips = _redirect_hop(
                current_url,
                website_policy,
                deadline,
                cancel_event,
            )
            if hop_error is not None:
                return hop_error, "", ""
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
            # The true type routes extracted text away from html_to_markdown.
            return None, pdf_text, "application/pdf"

        if not _is_text_candidate_content_type(content_type):
            m = re.match(r"[\w.+-]+/[\w.+-]+", content_type or "")
            safe_type = m.group(0) if m else "unknown type"
            return (
                f"(non-text content: {safe_type}, {len(raw_bytes)} bytes; not readable as text)",
                "",
                content_type,
            )

        if _has_binary_magic(raw_bytes):
            return (
                f"(binary content, {len(raw_bytes)} bytes; not readable as text)",
                "",
                content_type,
            )

        declared = resp.headers.get_content_charset()
        declared_codec = None
        try:
            if declared:
                declared_codec = codecs.lookup(declared).name
        except (LookupError, ValueError):
            declared = None
        bom_codec = next(
            (codec for bom, codec in _UNICODE_BOM_CODECS if raw_bytes.startswith(bom)),
            None,
        )
        fallback_codec = (
            bom_codec
            or _sniff_meta_charset(raw_bytes[:_META_CHARSET_SCAN_BYTES], content_type)
            or "utf-8"
        )
        try:
            raw_html = raw_bytes.decode(declared or fallback_codec, errors = "replace")
        except (LookupError, ValueError):
            # Lookup succeeds but decode fails for non-text codecs (base64, hex, zlib, undefined, idna).
            declared = None
            declared_codec = None
            raw_html = raw_bytes.decode(fallback_codec, errors = "replace")

        if _looks_binary(raw_html):
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


# Ambiguous tags are excluded: Markdown READMEs often open with them.
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
    """True only when the document ITSELF opens with HTML. Matches an HTML doctype or a leading
    document/structure tag after optional whitespace, not a mere substring, so a Markdown README
    with a fenced HTML example or tags further down stays Markdown. Also detects bare fragments
    with no doctype, so a page with a missing/wrong Content-Type is still converted."""
    probe = body.lstrip()[:256].lower()
    return bool(_HTML_LEADING_RE.match(probe))


# Only a real document opener, so a Markdown README starting with an HTML block is not
# collapsed by html_to_markdown.
_HTML_DOCUMENT_RE = re.compile(r"<(?:!doctype\s+html\b|/?(?:html|head|body)\b)")


def _looks_like_html_document(body: str) -> bool:
    """True only when the body opens as a full HTML document (e.g. a .html README)."""
    probe = body.lstrip()[:256].lower()
    return bool(_HTML_DOCUMENT_RE.match(probe))


def _loaded_context_tokens() -> int | None:
    """The active model's context window, or None when it cannot be read.

    Mirrors `research_runs._loaded_context_length` and `routes.inference._monitor_context_length`:
    llama.cpp first, then the orchestrator the API layer reads. Both branches are needed. A
    native/Transformers chat leaves `is_loaded` false, and stopping at that probe reported unknown,
    which kept the full 16,000-character cap and reproduced on small native models exactly the
    overflow this budget exists to prevent.

    The ML backends live in a worker subprocess, so the in-process singleton is unpopulated here and
    importing it pulls in the ML stack; peek at the orchestrator instead of constructing one. Every
    failure is unknown so a fetch is never blocked by not knowing.
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


def _request_context_tokens() -> int | None:
    """The request's window (an int, or None = unknowable: never probed), else the process probe. By type, not
    `is _UNSET_CONTEXT_TOKENS`: an `execute_tool` held across a reload stores the old sentinel (#11384)."""
    scoped = _REQUEST_CONTEXT_TOKENS.get()
    if scoped is None or isinstance(scoped, int):
        return scoped
    return _loaded_context_tokens()


def _result_char_budget(cap: int) -> int:
    """`cap`, lowered to what the serving window can actually hold. Shared by fetched pages and by
    terminal/python results, because the failure is the same: a fixed character cap has no
    relation to the loaded context, so on a small window one result fills most of it. That result
    lands in the NEWEST turn, which the fit protects, so compaction cannot drop the very thing
    that does not fit and the request goes irreducible. Measured live on a 5120-token window:
    7043 and 6684 token requests refused, both on the code tools, whose 16,000-character cap is
    about 4,000 tokens on its own."""
    ctx = _request_context_tokens()
    if not ctx:
        return cap
    # Clamped to `cap` on the way out so the floor never raises the configured cap.
    return min(cap, max(_MIN_PAGE_CHARS, int(ctx * 4 * _PAGE_CONTEXT_SHARE)))


def _tool_result_char_budget() -> int:
    """The terminal/python cap, sized to the window. See `_result_char_budget`."""
    return _result_char_budget(_MAX_OUTPUT_CHARS)


def _page_char_budget() -> int:
    """`_MAX_PAGE_CHARS`, lowered to what the serving window can actually hold.

    16,000 characters is roughly 4,000 tokens: fine on a 128k model, nonsensical on a 4,864-token
    one. Measured there, a single fetched page came back at 12,295 characters, the request went
    irreducible at 8,995 tokens against a 3,648-token budget with `latest_turn_role: "tool"`, and
    the user was advised to shorten a conversation consisting of one 11-token question. Nothing
    downstream can recover from it either: the fit protects the newest turn.

    Above roughly an 11k window this returns the old constant unchanged, so only the models that
    cannot afford a whole page are affected.
    """
    ctx = _request_context_tokens()
    if not ctx:
        return _MAX_PAGE_CHARS
    return max(_MIN_PAGE_CHARS, min(_MAX_PAGE_CHARS, int(ctx * 4 * _PAGE_CONTEXT_SHARE)))


def _request_result_room() -> int | None:
    """Tokens this result may add before the NEXT prompt is over budget. None when the caller could
    not say, and every cap then behaves exactly as it did before this existed: external
    providers, the hosted path and any tool loop that does not price its own conversation all
    take that leg."""
    room = _REQUEST_RESULT_BUDGET.get()
    if room is None:
        return None
    try:
        return max(0, int(room))
    except (TypeError, ValueError):
        return None


def _window_context_tokens() -> int | None:
    """The window this request is served by, or None when it cannot be read."""
    ctx = _request_context_tokens()
    return ctx if ctx else None


def _dense_prefix_chars(text: str, token_budget: float) -> int:
    """How many leading characters of `text` cost at most `token_budget` tokens.

    Four characters per token is an English rate. Measured with Qwen3, Llama 3.2 and tiktoken on
    real fetched pages, CJK prose runs 1.3-1.6 characters per token, and the percent-escaped links a
    CJK page is full of run 1.3-1.5: both are the same non-ASCII bytes, one spelled in ASCII.
    Charging them a token each, the rule `context_window.estimate_messages_tokens_dense` already
    uses, keeps the share the caller asked to reserve a share instead of the whole budget.

    One pass, so it costs nothing next to the fetch it sizes.
    """
    spent = 0.0
    index = 0
    length = len(text)
    while index < length:
        start = index
        if text[index] == "%" and _HEX_PAIR_RE.match(text, index + 1):
            spent += 3.0  # a percent-escaped byte; charge it like one non-ASCII byte
            index += 3
        else:
            spent += 1.0 if ord(text[index]) > 127 else 0.25
            index += 1
        # Cut on whole characters and escapes.
        if spent > token_budget:
            return start
    return length


# Probe as a user turn: some templates (Gemma-4) drop standalone tool messages, so their
# payload would go unpriced, and Mistral rejects most tool call ids.
_PROBE_ROLE = "user"

# Far past real text (densest measured is 128 chars/token); a template dropping content lands
# in the hundreds or infinity.
_MAX_PROBE_CHARS_PER_TOKEN = 256

# Counts are a pure function of (model, template, window, chunk) and cost two llama-server
# calls. Keyed on the server process: user extra args can override the chat template.
_PROBE_COUNT_CACHE: dict = {}

# Tool calls run in threads; the LRU touch and eviction are not atomic. Held only around dict
# work, never across a count; a duplicate measurement is harmless.
_PROBE_COUNT_LOCK = threading.Lock()

# Named so the two special cases cannot drift from a bare "".
_PROBE_BASELINE = ""

# One model at a time (a new identity clears it); ten times one result's worst case.
_PROBE_COUNT_CACHE_ENTRIES = 64

# Bounds held characters too: a large configured cap on a large window makes one prefix huge.
# The baseline is 0 chars, so it is never squeezed out.
_PROBE_COUNT_CACHE_CHARS = 1_000_000


def _probe_identity(llama, ctx: int):
    """A key that changes whenever a measured count could, or None to disable the cache. None is the
    safe answer: it costs round trips, it never returns a stale number."""
    try:
        # No process, no key: uncacheable backends keep paying round trips (the safe direction).
        pid = getattr(getattr(llama, "_process", None), "pid", None)
        if not isinstance(pid, int):
            return None
        key = (
            pid,
            ctx,
            getattr(llama, "model_identifier", None),
            getattr(llama, "_gguf_load_identity", None),
            getattr(llama, "_chat_template_override", None),
            # User-appended args, including a template override.
            tuple(getattr(llama, "_extra_args", None) or ()),
        )
        hash(key)
        return key
    except Exception:  # noqa: BLE001 -- an unreadable identity is "do not cache"
        return None


def _probe_cache(llama, ctx: int) -> dict:
    """The count cache for the model serving this request. A fresh per-call dict when the model has
    no identity, so the caller's code path is the same either way and an unidentifiable backend
    simply gets no reuse between calls."""
    identity = _probe_identity(llama, ctx)
    if identity is None:
        return {}
    with _PROBE_COUNT_LOCK:
        cache = _PROBE_COUNT_CACHE.get(identity)
        if cache is None:
            _PROBE_COUNT_CACHE.clear()
            cache = _PROBE_COUNT_CACHE[identity] = {}
    return cache


def _neutralized_for_prompt(chunk: str, llama) -> str:
    """``chunk`` as the outgoing request will carry it, rather than as it is in hand.

    The same sweep the request path applies, through the same helper and the backend's own markup
    profile, so the two cannot disagree about what a marker becomes. `_PROBE_ROLE` is a user role,
    which takes the full control rewrite a tool result takes rather than the boundary-only one an
    assistant turn takes.

    Best effort: a sweep that cannot run leaves the text as it was, which is the estimate this had
    before and never worse.
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
    """The tokenizer of the model serving this request, or None when there is not one. Same probe as
    `_loaded_context_tokens`: whatever can answer for the window can also price a string exactly,
    and `llama_cpp` already hands this same counter to the RAG admission check for exactly this
    reason. Gated on the backend's own window matching the one the budget was sized against, so a
    resident GGUF never prices a request that a different model (native, or an external endpoint)
    is actually answering."""
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
    # Both only gate the strict attempt, never a returned value.
    retained = bool(_probe_identity(llama, ctx))
    template_down: list[bool] = []

    def _remember(chunk: str, value: int) -> None:
        """Hold `value` for `chunk`, evicting least-recently-used entries to stay in bounds.

        Refusing new entries once full was worse than not caching at all. Most tool results are
        one-offs, so the first `_PROBE_COUNT_CACHE_ENTRIES` distinct prefixes froze the cache on
        text that would never be asked about again, and because the baseline is only priced when a
        count comes in OVER budget, a process that handled 64 results that FIT first locked it out
        for good. Measured: after 64 English results, every later dense result paid 4 counter calls
        (8 HTTP) again, exactly the merge base's cost, for the life of the process.

        So evict, and pin the baseline: it is 0 characters, it is the same number for every result
        this process truncates, and it is the one entry a bounded cache most needs.
        """
        if len(chunk) > _PROBE_COUNT_CACHE_CHARS:
            return
        with _PROBE_COUNT_LOCK:
            held = sum(map(len, cache))
            while len(cache) >= _PROBE_COUNT_CACHE_ENTRIES or (
                held + len(chunk) > _PROBE_COUNT_CACHE_CHARS
            ):
                # list() and pop(..., None) tolerate concurrent inserts and evictions.
                victim = next((key for key in list(cache) if key != _PROBE_BASELINE), None)
                if victim is None:
                    return
                held -= len(victim)
                cache.pop(victim, None)
            cache[chunk] = value

    def _rendered(chunk: str):
        # Price the neutralized text: control-markup sweeping (#7066) inflates token counts.
        chunk = _neutralized_for_prompt(chunk, llama)
        with _PROBE_COUNT_LOCK:
            hit = cache.get(chunk)
            if hit is not None and chunk != _PROBE_BASELINE:
                if cache.pop(chunk, None) is not None:
                    cache[chunk] = hit
        if hit is not None:
            return hit
        # Strict so only template-rendered counts are cached; the non-strict fallback drops role
        # markers. Asked at most once per counter.
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
        # The fallback count is used (it still tokenizes real bytes) but not retained.
        if value is not None and rendered:
            _remember(chunk, value)
        return value

    # Empty-turn baseline: a template rendering no content shows as a count that does not move.
    # Left in the total (errs smaller).
    baseline: list[int] = []

    def _framing() -> int:
        if not baseline:
            baseline.append(_rendered(_PROBE_BASELINE) or 0)
        return baseline[0]

    def _count(chunk: str, token_budget: float = 0.0):
        """Tokens for `chunk`, or None when the count did not measure it. `token_budget` is an
        optimisation and nothing more. A count within budget and a count the guard rejects both
        make the caller return its own estimate unchanged, so when `spent` fits, the baseline
        that separates those two paths cannot change the answer and is not priced, which is why
        an English result costs one round trip rather than two. The default of 0 means no budget,
        so the guard always runs."""
        spent = _rendered(chunk)
        if spent is None:
            return None
        if spent <= token_budget:
            return spent
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


# English fits in one pass, base64 two, mixed three; bounded since each pass is a round trip.
_EXACT_FIT_PASSES = 5


def _exact_prefix_chars(
    text: str,
    chars: int,
    token_budget: float,
    ctx: int,
    floor: int | None = None,
) -> int:
    """`chars`, shrunk until the prefix really costs `token_budget`. Never grown.

    The estimate below charges every ASCII character a flat 0.25 tokens, which is an English rate
    and wrong in the same direction for the ASCII the code tools print most: measured with Qwen3-4B
    and Llama-3.2 on a 5,120-token window, where the character cap admits 7,168 characters against a
    1,792-token share, `base64 payload.bin` came back at 5,361 tokens, `hexdump -C` at 5,540 and
    `sha256sum *` at 5,109 -- 105-108% of the WHOLE window, in the newest turn, which the fit
    protects. A four-message thread was refused irreducible at 5,475 tokens against a 3,840-token
    prompt budget. No character rule closes that: the same rule that charges a 76-character base64
    line its real 57 tokens charges English prose 40% more than it costs. So when a tokenizer is
    serving the request, ask it; when none is, keep the estimate exactly as it was.
    """
    # The caller's floor: the 2,000-char comfort floor can overflow a nearly full thread.
    if floor is None:
        floor = _MIN_PAGE_CHARS
    # At or below the caller's floor nothing measured could change the answer.
    if chars <= floor:
        return chars
    counter = _loaded_token_counter(ctx)
    if counter is None:
        return chars
    # Returns a measured fit, the caller's estimate, or the floor, never an unmeasured shrink:
    # cutting prose off dense-then-English output raises the remaining density.
    previous = None
    for _ in range(_EXACT_FIT_PASSES):
        # A count that already fits can skip pricing the baseline. See `_count`.
        spent = counter(text[:chars], token_budget)
        if spent is None:
            return chars
        if spent <= token_budget:
            return chars
        fitted = int(chars * token_budget / spent)
        if previous is not None:
            # Two measurements price the tail that was cut; take the smaller step.
            prior_chars, prior_spent = previous
            per_char = (prior_spent - spent) / (prior_chars - chars)
            if per_char > 0:
                fitted = min(fitted, chars - int((spent - token_budget) / per_char))
        previous = (chars, spent)
        if fitted <= floor:
            return floor
        chars = min(fitted, chars - 1)
    # Out of passes with the last shrink unmeasured: fall back to the floor.
    return floor


def _can_measure_tokens(ctx: int, text: str) -> bool:
    """Whether this request's tokens can really be counted, not merely whether a counter is exposed.

    `_loaded_token_counter` returns a callable that answers None whenever the probe does not come
    back with a number: `/apply-template` failing, or a chat template that drops the probe role, or
    a backend that stopped serving between the check and the call. `_exact_prefix_chars` then hands
    back the caller's estimate untouched, which charges plain ASCII the English four characters per
    token; base64, minified JSON and hashes run nearer two, so a room that was never halved is spent
    about twice over. A counter that cannot measure has to be treated exactly like a counter that is
    not there.

    Probed on this text's own opening rather than a constant, so a template that refuses some
    content and not other content is judged on what is actually being sized.
    """
    counter = _loaded_token_counter(ctx)
    if counter is None:
        return False
    return counter(text[:_MEASURABILITY_PROBE_CHARS] or "x") is not None


_MEASURABILITY_PROBE_CHARS = 64


def _text_token_cost(text: str, ctx: int) -> float:
    """What ``text`` really costs, measured when the serving model can measure it. The estimate is
    the inverse of `_dense_prefix_chars`: ASCII at the English four characters per token,
    everything else at one. Doubled when nothing can check it, for the same reason
    `_UNMEASURED_ROOM_MARGIN` halves a room that cannot be measured."""
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
    # A counter that could not answer is absent; its presence proves nothing.
    ascii_chars = len(text.encode("ascii", "ignore"))
    estimate = ascii_chars * 0.25 + (len(text) - ascii_chars)
    return estimate / _UNMEASURED_ROOM_MARGIN


def _dense_char_limit(
    text: str,
    max_chars: int,
    reserve_tokens: float = 0.0,
) -> int:
    """`max_chars`, lowered when `text` tokenises denser than four characters per token. Without
    this the window-derived caps above reserve their share only for English. On the 4,864-token
    window this PR was measured against, the 6,809-character page budget is 35% of the window in
    English and 3,800-4,500 real tokens of a Chinese or Japanese page: 80-95% of the whole prompt
    budget, in the newest turn, which the fit protects. That is the same irreducible refusal the
    budget exists to prevent."""
    ctx = _window_context_tokens()
    room = _request_result_room()
    if room is None and (not ctx or len(text) <= _MIN_PAGE_CHARS):
        return max(0, max_chars - int(reserve_tokens * 4))
    if room is not None and not _can_measure_tokens(ctx or 0, text):
        # Nothing here can measure this model's tokens, and 4 chars/token undercharges dense ASCII;
        # halve so a wrong estimate errs short.
        room = int(room * _UNMEASURED_ROOM_MARGIN)
    # A float so English text lands exactly on the budget. Unknown window: the room is the answer.
    share = float(ctx * _PAGE_CONTEXT_SHARE) if ctx else float(room)
    # Reserve comes off the token budget: dense appended text costs more than its length.
    share = max(0.0, share - reserve_tokens)
    if room is not None:
        room = max(0, int(room - reserve_tokens))
        # The share does not fall as the thread fills; the room does.
        share = min(share, float(room))
    floor = _MIN_PAGE_CHARS
    if room is not None:
        # The floor yields to the room, measured, bottomed at one character.
        room_chars = _dense_prefix_chars(text, float(room))
        if not ctx:
            return min(max_chars, room_chars)
        # Bottomed at zero so a full thread gets the cheap stub, not the ~90-token notice.
        floor = min(floor, _exact_prefix_chars(text, room_chars, float(room), ctx, 0))
    fitted = _dense_prefix_chars(text, share)
    # Measured when possible; the floor goes with it or a nearly full thread gets 2,000 chars.
    fitted = _exact_prefix_chars(text, min(fitted, max_chars), share, ctx, floor)
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
    # Resolved per call: an import-time default freezes before any model loads.
    max_chars: int | None = None,
    timeout: int = 30,
    cancel_event = None,
    website_policy: dict | None = None,
) -> str:
    """Fetch a URL and return readable text content. HTML responses are converted to Markdown with a
    main-content heuristic (``<article>``/``<main>`` scoping, hidden-element and boilerplate
    stripping); non-HTML text responses are returned as-is. GitHub repo root pages are rewritten
    to the README API so the model reads the README instead of the repo page's UI chrome. Blocks
    private/loopback/link-local targets (SSRF protection) and caps the download size."""
    if max_chars is None:
        max_chars = _page_char_budget()
    # One deadline for the whole fetch so the HTML fallback does not get a fresh timeout.
    deadline = None if timeout is None else time.monotonic() + timeout
    from .web_access_policy import check_url_access

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
        # A 200 README body is authoritative even when HTML; fall back only on failure.
        if err is None and body.strip():
            readme_body = body
            if _looks_like_html_document(body):
                from ._html_to_md import html_to_markdown
                converted = html_to_markdown(body, main_content = True, max_span_chars = max_chars // 2)
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

    # Sniff the body when Content-Type is missing or wrong.
    is_html = "html" in content_type or _looks_like_html(body)
    if not is_html:
        # Converting plain text through the HTML renderer would collapse whitespace.
        return _truncate_page_text(body.strip(), max_chars)

    # Convert HTML to Markdown with the builtin converter (no external deps).
    from ._html_to_md import SiteLinks, html_to_markdown

    site_links = SiteLinks(url)
    # generated span cells get half the window budget, as html_to_markdown's own cap does for 16K
    text = html_to_markdown(
        body, main_content = True, site_links = site_links, max_span_chars = max_chars // 2
    )
    # a page that fits keeps same-site links so the model can follow them.
    if text and len(text) <= max_chars and len(text) <= _dense_char_limit(text, max_chars):
        return text
    return _truncate_page_text(site_links.strip(text), max_chars)


def _search_failure_message(exc: BaseException, timeout: int) -> str:
    """Turn a ddgs exception into text the model and the UI can act on.

    ddgs raises for an empty sweep as well as for refusals, so an unclassified ``Search failed:
    {exc}`` reports nothing matched and every engine throttled us the same way. Matched by class
    name because ddgs is imported lazily and tests stub the module.

    The RatelimitException arm is forward-looking: ddgs 9.14.4 defines the class but raises it
    nowhere, and no engine inspects the status code, so a throttled sweep parses to zero items and
    arrives here as the empty-sweep DDGSException instead.
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
    if name == "DDGSException" and _DDGS_EMPTY_SWEEP in str(exc):
        return EMPTY_SEARCH_RESULTS[0]
    return f"Search failed: {exc}"


def _resolve_engine_tiers(text_engines) -> list:
    """``_SEARCH_ENGINE_TIERS`` reduced to the engines this ddgs actually has, in tier order.

    Naming an engine the registry lacks is not an error: ddgs 9.8.0 raises ``KeyError`` on the first
    unknown name and silently re-runs the request as ``auto``, the Yandex fan-out this exists to
    prevent, and tier 1 trips it there because 9.8.0 ships no ``startpage``. An empty tier is dropped
    for the same reason.
    """
    engines = text_engines or {}
    resolved = []
    for tier in _SEARCH_ENGINE_TIERS:
        live = [
            name
            for name in tier
            if engines.get(name) is not None and not getattr(engines.get(name), "disabled", False)
        ]
        if live:
            resolved.append(",".join(live))
    return resolved


def _class_token(name: str) -> str:
    return f"contains(concat(' ', normalize-space(@class), ' '), ' {name} ')"


# Each section.algo leaves a div unclosed, so lxml nests every later result inside it: match the
# nearest section, and take the next s-desc before the next title (a descendant search takes them all).
_YAHOO_SECTION_TITLES = (
    f"//a[{_class_token('s-title')}][ancestor::section[1][{_class_token('algo')}]]"
)
_YAHOO_SECTION_SNIPPET = f"following::*[self::p[{_class_token('s-desc')}] or self::a[{_class_token('s-title')}]][1][self::p]"


def _install_yahoo_layout_parser(text_engines) -> None:
    """Parse Yahoo's ``section.algo`` pages, which ddgs 9.8.0-9.16.0 (``div.relsrch`` only) read as
    empty. ddgs builds engines from this registry by name; ``relsrch`` pages keep ddgs's parser."""
    yahoo = (text_engines or {}).get("yahoo")
    if not isinstance(yahoo, type) or getattr(yahoo, "_parses_section_layout", False):
        return

    class _Yahoo(yahoo):
        _parses_section_layout = True

        def extract_results(self, html_text):
            results = super().extract_results(html_text)
            if results:
                return results
            tree = self.extract_tree(self.pre_process_html(html_text))
            for link in tree.xpath(_YAHOO_SECTION_TITLES):
                # TextResult strips tags and collapses whitespace on assignment.
                result = self.result_type()
                result.title = link.get("aria-label") or "".join(
                    link.xpath(f".//text()[not(ancestor::span[{_class_token('title-url')}])]")
                )
                result.href = link.get("href") or ""
                snippet = link.xpath(_YAHOO_SECTION_SNIPPET)
                if snippet:
                    result.body = snippet[0].xpath("string()")
                results.append(result)
            return results

    text_engines["yahoo"] = _Yahoo


def _is_connection_reset(exc) -> bool:
    return any(marker in f"{type(exc).__name__}: {exc}".lower() for marker in _DDGS_RESET_MARKERS)


def _ddgs_http1_replay(args, kwargs, config):
    import httpx

    verify = config["verify"]
    if isinstance(verify, str):
        verify = ssl.create_default_context(cafile = verify)
    with httpx.Client(
        headers = config["headers"],
        cookies = config["cookies"],
        proxy = config["proxy"],
        timeout = config["timeout"],
        verify = verify,
        follow_redirects = config["follow_redirects"],
        http1 = True,
        http2 = False,
    ) as client:
        resp = client.request(*args, **kwargs)
        resp.read()
        return resp


def _install_ddgs_http1_retry() -> None:
    """Replay a ddgs request once over plain HTTP/1.1 after a connection reset (#12638): primp has
    no HTTP/1.1-only mode. Successful requests are untouched; wraps each class once; never raises."""
    try:
        import inspect

        from ddgs import http_client
        from ddgs.exceptions import DDGSException

        def _wrapper(response_cls):
            # ddgs 9.14's primp Response wraps the raw response; 9.8.0's and HttpClient2's take fields.
            if "status_code" not in inspect.signature(response_cls).parameters:
                return response_cls
            return lambda resp: response_cls(
                status_code = resp.status_code, content = resp.content, text = resp.text
            )

        targets = [(http_client.HttpClient, _wrapper(http_client.Response), True)]
        try:
            from ddgs import http_client2
        except ImportError:
            http_client2 = None
        if http_client2 is not None and hasattr(http_client2, "HttpClient2"):
            # HttpClient2 does not follow redirects; primp does.
            targets.append((http_client2.HttpClient2, _wrapper(http_client2.Response), False))

        with _DDGS_HTTP1_RETRY_LOCK:
            for cls, wrap, follow_redirects in targets:
                if cls.__dict__.get("_unsloth_http1_retry"):
                    continue
                _wrap_ddgs_client(
                    cls, wrap, follow_redirects, inspect.signature(cls.__init__), DDGSException
                )
    except Exception:  # noqa: BLE001 - the retry is a hardening layer, never a reason to fail a search
        logger.debug("ddgs HTTP/1.1 retry not installed", exc_info = True)


def _wrap_ddgs_client(cls, wrap, follow_redirects, signature, ddgs_exception) -> None:
    orig_init, orig_request = cls.__init__, cls.request

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            config = dict(bound.arguments)
            config.pop("self", None)  # no client -> config -> client cycle
            self._unsloth_http1_config = config
        except TypeError:
            pass

    @functools.wraps(orig_request)
    def request(self, *args, **kwargs):
        start = time.monotonic()
        try:
            return orig_request(self, *args, **kwargs)
        except Exception as exc:
            config = getattr(self, "_unsloth_http1_config", None)
            if config is None or not _is_connection_reset(exc):
                raise
            timeout = config.get("timeout")
            if timeout:
                timeout -= time.monotonic() - start
                if timeout <= 0:
                    raise
            client = getattr(self, "client", None)
            # httpx's jar only: primp 0.15 (ddgs 9.8.0) get_cookies aborts the process on a miss.
            cookies = getattr(client, "cookies", None)
            # httpx may lack primp's zstd decoder, so let it pick accept-encoding.
            session = getattr(client, "headers", None) or {}
            headers = {k: v for k, v in dict(session).items() if k.lower() != "accept-encoding"}
            replay = {
                "headers": headers,
                "cookies": cookies,
                "proxy": config.get("proxy"),
                "timeout": timeout,
                "verify": config.get("verify", True),
                "follow_redirects": follow_redirects,
            }
            try:
                return wrap(_ddgs_http1_replay(args, kwargs, replay))
            except Exception as retry_exc:
                raise ddgs_exception(
                    f"{exc}; HTTP/1.1 retry failed: {type(retry_exc).__name__}: {retry_exc}"
                ) from retry_exc

    cls.__init__, cls.request = __init__, request
    cls._unsloth_http1_retry = True


def _image_search_or_none(subjects: list, timeout, cancel_event, website_policy) -> "str | None":
    """``_image_search`` that reports a failure as None instead of raising. Every caller sits inside
    ``_web_search``'s own ``except``, which would turn a raise into Search failed: ... and throw
    away the text results the search had already found. A picture is a garnish: it must not
    become the answer."""
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
    """``empty_text`` plus the pictures the model asked for by name, if any. An ``image_queries``
    call is an explicit request, and it succeeds on its own when sent without a query, so
    returning the bare No results found. because the TEXT sweep came back empty dropped images
    that were there to be had. Only the named subjects are looked up here; the per-query image
    pile has no answer to garnish."""
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


def _wikipedia_search(query, max_results, timeout, deadline, cancel_event, website_policy):
    """search English Wikipedia independently of ddgs through the guarded HTTP fetcher."""
    # ddgs uses a one-result Wikipedia lookup, so full-text search recovers misses and failures.
    from html import unescape

    params = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": min(max_results, 50),
            "srnamespace": 0,
        }
    )
    error, body, _ = _fetch_url_raw(
        "https://en.wikipedia.org/w/api.php?" + params,
        timeout = timeout,
        deadline = deadline,
        cancel_event = cancel_event,
        website_policy = website_policy,
        raw_bytes_max = 1024 * 1024,
        extra_headers = {"User-Agent": "UnslothStudio/1.0 (https://github.com/unslothai/unsloth)"},
    )
    if error:
        raise RuntimeError(error)
    payload = json.loads(body)
    if "error" in payload:
        raise RuntimeError("Wikipedia search API returned an error")
    return [
        {
            "title": item["title"],
            "href": "https://en.wikipedia.org/wiki/"
            + urllib.parse.quote(item["title"].replace(" ", "_"), safe = ""),
            "body": unescape(re.sub(r"<[^>]+>", "", item.get("snippet", ""))),
        }
        for item in payload["query"]["search"]
        if isinstance(item, dict) and isinstance(item.get("title"), str) and item["title"].strip()
    ]


def _usable_search_results(results, website_policy):
    from .web_access_policy import check_url_access
    return [
        r
        for r in results
        if isinstance(r, dict)
        and check_url_access(str(r.get("href") or "").strip(), website_policy)[0]
    ]


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
    """search approved tiers, fetch a URL, or return registered ``[[img:<id>]]`` images alone."""
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
        # guard here because execute_tool requires a string result and this is outside the try.
        found = _image_search_or_none(subjects, timeout, cancel_event, website_policy)
        if found is None:
            return "No images found for: " + ", ".join(subjects)
        return found

    if not query or not query.strip():
        return "No query provided."
    # DDGS.text() is blocking, so cancellation is checked before and after the call.
    if cancel_event is not None and cancel_event.is_set():
        return "Search cancelled."
    try:
        from .web_access_policy import check_url_access, scope_search_query

        effective_query = scope_search_query(query, website_policy)
        # overfetch for allowed hits below blocked ones; normalized policy remains truthy.
        restricted = any(
            (website_policy or {}).get(key) for key in ("allowedDomains", "blockedDomains")
        )
        wanted = max_results * _POLICY_OVERFETCH if restricted else max_results
        # bound fallback even if importing or resolving ddgs fails before its normal budget starts.
        deadline = time.monotonic() + timeout if timeout else None
        client, results, last_error = None, [], None
        rejected_results = False
        wikipedia_fallback = False
        try:
            from ddgs import DDGS
            from ddgs.engines import ENGINES

            text_engines = ENGINES.get("text") or {}
            _install_yahoo_layout_parser(text_engines)
            _install_ddgs_http1_retry()
            engine_tiers = _resolve_engine_tiers(text_engines)
            if not engine_tiers:
                raise RuntimeError("no approved search engine is available.")
            # reset after setup to keep the primary budget; the earlier deadline bounds fallback.
            deadline = time.monotonic() + timeout if timeout else None
            client = DDGS(timeout = timeout)
            # DDGS uses per-client timeouts; tiers share one budget and images reuse the client.
            for backend in engine_tiers:
                if cancel_event is not None and cancel_event.is_set():
                    return "Search cancelled."
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    client = DDGS(timeout = remaining)
                try:
                    candidates = client.text(effective_query, max_results = wanted, backend = backend)
                    results = _usable_search_results(candidates, website_policy)
                    rejected_results = rejected_results or bool(candidates and not results)
                except Exception as exc:  # noqa: BLE001 - try the next tier before classifying the failure
                    last_error = exc
                    continue
                if results:
                    break
        except Exception as exc:
            last_error = exc
        if cancel_event is not None and cancel_event.is_set():
            return "Search cancelled."
        if not results:
            remaining = deadline - time.monotonic() if deadline else 5.0
            allowed, _, _ = check_url_access("https://en.wikipedia.org/w/api.php", website_policy)
            if allowed and remaining > 0:
                try:
                    fallback_timeout = min(remaining, 5.0)
                    fallback_deadline = time.monotonic() + fallback_timeout
                    if deadline is not None:
                        fallback_deadline = min(fallback_deadline, deadline)
                    results = _usable_search_results(
                        _wikipedia_search(
                            query,
                            wanted,
                            fallback_timeout,
                            fallback_deadline,
                            cancel_event,
                            website_policy,
                        ),
                        website_policy,
                    )
                    wikipedia_fallback = bool(results)
                except Exception:
                    logger.debug("Independent Wikipedia search failed", exc_info = True)
            if cancel_event is not None and cancel_event.is_set():
                return "Search cancelled."
        # blocked results take precedence over earlier tier exceptions.
        if not results and last_error is not None and not rejected_results:
            raise last_error
        if not results:
            return _empty_result_with_requested_images(
                EMPTY_SEARCH_RESULTS[1]
                if rejected_results and restricted
                else EMPTY_SEARCH_RESULTS[0],
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
            title = " ".join(str(r.get("title") or href).split())
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
        if wikipedia_fallback:
            text = (
                "General web search was unavailable or returned no usable results. "
                "These are Wikipedia-only encyclopedia results, not current web coverage.\n\n"
                + text
            )
        text += (
            "\n\n---\n\nIMPORTANT: These are only short snippets. "
            "To get the full page content, call web_search with "
            'the url parameter (e.g. {"url": "<URL>"}).'
        )
        if include_images and subjects:
            # named subjects require one image each rather than a generic image batch.
            found = _image_search_or_none(subjects, timeout, cancel_event, website_policy)
            if found is not None:
                text += "\n\n---\n\n" + found
        elif include_images and not wikipedia_fallback:
            text += _web_search_images_suffix(
                client,
                effective_query,
                wanted,
                cancel_event,
                website_policy,
            )
        elif subjects:
            # replayed history must retain the disabled-search reminder.
            text += "\n\n---\n\n" + IMAGE_SEARCH_DISABLED
        return text
    except Exception as e:
        failure = _search_failure_message(e, timeout)
        # ddgs raises on an empty sweep; attach requested images only to empty results, not genuine errors.
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
    _install_ddgs_http1_retry()
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
    # Read before the sweep so a clear-all during it invalidates new entries.
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


# `urllib.request`/`urllib3` share `urllib`; `http.client`/`httpx` share `http`.
_NETWORK_ROOT_NAMES = frozenset(
    {
        "socket",
        "_socket",
        "urllib",
        "urllib3",
        "http",
        "httpx",
        "requests",
        "aiohttp",
        "paramiko",
        "fabric",
        "asyncssh",
    }
)


def _network_candidates_possible(nodes) -> bool:
    """Whether ANY call in this tree could resolve to a network function.

    The alias maps, the literal-value map, the shadow bookkeeping and the second visitor pass all
    exist to answer questions about a recognised egress call, and every route to one names a
    network module: an import of it, an import from it, or the module written out at the call
    site, which is a `Name` either way. A tree with none of those has no candidate to resolve, so
    the screen can skip all of it and still reach the same verdict. Ordinary tool code -- pandas,
    matplotlib, plain arithmetic -- takes that path.

    Read off the node list the other classifiers already build and cache for this tree, so it
    costs one pass over a list rather than a walk. Deliberately generous: a local variable that
    happens to be called `socket` turns the full path back on, which costs time and never a
    verdict.
    """
    for node in nodes:
        if isinstance(node, ast.Name):
            if node.id in _NETWORK_ROOT_NAMES:
                return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.partition(".")[0] in _NETWORK_ROOT_NAMES:
                    return True
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").partition(".")[0] in _NETWORK_ROOT_NAMES:
                return True
    return False


def _check_signal_escape_patterns(code: str):
    """Check for patterns that could escape signal-based timeouts. Returns (safe: bool, details:
    dict). Vendored from unsloth_zoo.rl_environments to avoid importing unsloth_zoo (needs GPU
    drivers; fails on Apple Silicon)."""
    tree, _parse_error = _parse_python(code)
    if _parse_error is not None:
        e = _parse_error
        return False, {
            "error": f"SyntaxError: {e}",
            "signal_tampering": [],
            "exception_catching": [],
            "warnings": [],
        }

    # Every route to a recognised egress call spells the module's name, so one text pass decides
    # whether the alias machinery can matter.
    network_possible = _network_candidates_possible(_tree_nodes(tree))

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

            shell_func = None
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name):
                    if func.value.id in self.os_aliases:
                        shell_func = f"os.{func.attr}"
                    elif func.value.id in self.subprocess_aliases:
                        shell_func = f"subprocess.{func.attr}"
            elif isinstance(func, ast.Name):
                shell_func = self.shell_exec_aliases.get(func.id)

            if shell_func and shell_func in _SHELL_EXEC_FUNCS:
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
                    # Any non-literal-False shell= is treated as True.
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
                # `except Exception` cannot catch SystemExit/KeyboardInterrupt, so it is fine.
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

    # An unreadable host is untrusted: nothing downstream screens python-tool code again.
    network_calls: list[dict] = []
    sensitive_file_reads: list[dict] = []
    _NETWORK_FQ_PREFIXES = (
        "socket.socket",
        "socket.create_connection",
        "socket.getaddrinfo",
        "_socket.socket",
        "_socket.SocketType",
        "_socket.getaddrinfo",
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
    # The modules an alias is resolved back to.
    _NETWORK_MODULES = frozenset(
        {
            "socket",
            "_socket",
            "urllib.request",
            "urllib3",
            "urllib3.connection",
            "urllib3.connectionpool",
            "urllib3.poolmanager",
            "urllib3.util.connection",
            "urllib3.contrib.socks",
            "http.client",
            "requests",
            "requests.api",
            "requests.sessions",
            "httpx",
            "aiohttp",
            "aiohttp.client",
            "paramiko",
            "paramiko.client",
            "paramiko.transport",
            "fabric",
            "fabric.connection",
            "asyncssh",
            "asyncssh.connection",
        }
    )
    _HTTP_VERBS = ("get", "post", "put", "delete", "patch", "head", "options")
    # Calls whose first positional is the host or URL.
    _NETWORK_URL_ARG0_FQ = frozenset(
        {
            "socket.create_connection",
            "socket.getaddrinfo",
            "_socket.getaddrinfo",
            "urllib.request.urlopen",
            "urllib.request.urlretrieve",
            "http.client.HTTPConnection",
            "http.client.HTTPSConnection",
            *(
                f"{module}.{verb}"
                for module in ("requests", "requests.api", "httpx")
                for verb in _HTTP_VERBS
            ),
        }
    )
    _HOST_ARG_ROOTS = ("socket.", "_socket.", "http.client.")
    _NETWORK_DESTINATION_ARG = {
        fq: (
            0,
            ("url", "fullurl", "host", "address"),
            "host" if fq.startswith(_HOST_ARG_ROOTS) else "url",
        )
        for fq in _NETWORK_URL_ARG0_FQ
    }
    # `urllib3.request(method, url)` puts the URL second (urllib3 2.8.0).
    _NETWORK_DESTINATION_ARG.update(
        {
            f"{module}.request": (1, ("url",), "url")
            for module in (
                "requests",
                "requests.api",
                "httpx",
                "urllib3",
                "aiohttp",
                "aiohttp.client",
            )
        }
    )
    _NETWORK_DESTINATION_ARG["httpx.stream"] = (1, ("url",), "url")
    _VERB_CLIENTS = (
        "requests.Session",
        "requests.sessions.Session",
        "requests.session",
        "requests.sessions.session",
        "httpx.Client",
        "httpx.AsyncClient",
        "aiohttp.ClientSession",
        "aiohttp.client.ClientSession",
    )
    _POOL_CLIENTS = (
        "urllib3.PoolManager",
        "urllib3.ProxyManager",
        "urllib3.poolmanager.PoolManager",
        "urllib3.poolmanager.ProxyManager",
        "urllib3.proxy_from_url",
        "urllib3.poolmanager.proxy_from_url",
        "urllib3.contrib.socks.SOCKSProxyManager",
    )
    _SOCKET_TYPES = ("socket.socket", "socket.SocketType", "_socket.socket", "_socket.SocketType")
    _SOCKET_CLIENTS = (
        *_SOCKET_TYPES,
        "paramiko.SSHClient",
        "paramiko.client.SSHClient",
    )
    _OPENER_CLIENTS = ("urllib.request.build_opener", "urllib.request.OpenerDirector")
    _CLIENT_CLASSES = frozenset(
        (*_VERB_CLIENTS, *_POOL_CLIENTS, *_SOCKET_CLIENTS, *_OPENER_CLIENTS)
    )
    _NETWORK_DESTINATION_ARG.update(
        {
            **{
                f"{client}.{verb}": (0, ("url",), "url")
                for client in _VERB_CLIENTS
                for verb in _HTTP_VERBS
            },
            **{
                f"{client}.request": (1, ("url",), "url")
                for client in (*_VERB_CLIENTS, *_POOL_CLIENTS)
            },
            **{
                f"{client}.stream": (1, ("url",), "url")
                for client in ("httpx.Client", "httpx.AsyncClient")
            },
            **{
                f"{client}.ws_connect": (0, ("url",), "url")
                for client in ("aiohttp.ClientSession", "aiohttp.client.ClientSession")
            },
            **{
                client: (None, ("base_url",), "url")
                for client in ("httpx.Client", "httpx.AsyncClient")
            },
            **{
                client: (0, ("base_url",), "url")
                for client in ("aiohttp.ClientSession", "aiohttp.client.ClientSession")
            },
            **{
                f"{client}.send": (0, ("request",), "url")
                for client in (
                    "httpx.Client",
                    "httpx.AsyncClient",
                    "requests.Session",
                    "requests.sessions.Session",
                )
            },
            **{
                f"{client}.{method}": (1, ("url",), "url")
                for client in _POOL_CLIENTS
                for method in ("urlopen", "request_encode_url", "request_encode_body")
            },
            **{f"{client}.connection_from_url": (0, ("url",), "url") for client in _POOL_CLIENTS},
            **{
                f"{client}.connection_from_host": (0, ("host",), "host") for client in _POOL_CLIENTS
            },
            **{
                f"{module}.{factory}": (0, ("url",), "url")
                for factory, defined_in in (
                    ("connection_from_url", "urllib3.connectionpool"),
                    ("proxy_from_url", "urllib3.poolmanager"),
                )
                for module in ("urllib3", defined_in)
            },
            **{
                f"{module}.ProxyManager": (0, ("proxy_url",), "url")
                for module in ("urllib3", "urllib3.poolmanager")
            },
            "urllib3.contrib.socks.SOCKSProxyManager": (0, ("proxy_url",), "url"),
            **{
                f"httpx.{transport}": (None, ("proxy",), "proxy")
                for transport in ("HTTPTransport", "AsyncHTTPTransport")
            },
            **{
                f"{module}.{pool}": (0, ("host",), "host")
                for module in ("urllib3", "urllib3.connectionpool")
                for pool in ("HTTPConnectionPool", "HTTPSConnectionPool")
            },
            **{
                f"urllib3.connection.{conn}": (0, ("host",), "host")
                for conn in ("HTTPConnection", "HTTPSConnection")
            },
            "urllib3.util.connection.create_connection": (0, ("address",), "host"),
            **{
                f"{sock}.{m}": (0, ("address",), "host")
                for sock in _SOCKET_TYPES
                for m in ("connect", "connect_ex")
            },
            **{f"{opener}.open": (0, ("fullurl",), "url") for opener in _OPENER_CLIENTS},
            "urllib.request.ProxyHandler": (None, (), "proxy"),
            # sendto's int flags at index 1 read as unreadable and fail closed.
            **{f"{sock}.sendto": (1, (), "host") for sock in _SOCKET_TYPES},
            **{f"{sock}.sendmsg": (3, (), "host") for sock in _SOCKET_TYPES},
            **{
                f"{client}.connect": (0, ("hostname", "host"), "host")
                for client in ("paramiko.SSHClient", "paramiko.client.SSHClient")
            },
            **{
                f"{module}.Transport": (0, ("sock",), "host")
                for module in ("paramiko", "paramiko.transport")
            },
            **{
                f"{module}.Connection": (0, ("host",), "host")
                for module in ("fabric", "fabric.connection")
            },
            **{
                f"{module}.{fn}": (index, ("host",), "host")
                for module in ("asyncssh", "asyncssh.connection")
                for fn, index in (
                    ("connect", 0),
                    ("connect_reverse", 0),
                    ("create_connection", 1),
                )
            },
            **{
                f"{module}.SSHClientConnectionOptions": (None, (), "host")
                for module in ("asyncssh", "asyncssh.connection")
            },
        }
    )
    _NETWORK_FQ_PREFIXES = _NETWORK_FQ_PREFIXES + tuple(
        fq
        for fq in sorted(_NETWORK_DESTINATION_ARG)
        if not any(fq.startswith(p) for p in _NETWORK_FQ_PREFIXES)
    )
    _NETWORK_ROOTS = frozenset(p.partition(".")[0] for p in _NETWORK_FQ_PREFIXES)
    _PROXY_KEYWORDS = ("proxy", "proxies")
    _DESTINATION_ATTRS = {
        "proxies": frozenset(
            (
                "requests.Session",
                "requests.sessions.Session",
                "requests.session",
                "requests.sessions.session",
            )
        ),
        "base_url": frozenset(("httpx.Client", "httpx.AsyncClient")),
    }
    _UPLOAD_HTTP_METHODS = (
        *(
            f"{owner}.{verb}"
            for owner in ("requests", "requests.api", "httpx", *_VERB_CLIENTS)
            for verb in ("post", "put", "patch", "delete", "request")
        ),
        "aiohttp.request",
        "aiohttp.client.request",
        *(f"{owner}.stream" for owner in ("httpx", "httpx.Client", "httpx.AsyncClient")),
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
    _TRUSTED_PUBLIC_HOST_LITERALS = frozenset(
        {
            "www.google.com",
            "google.com",
            "www.bing.com",
            "bing.com",
            "duckduckgo.com",
            "html.duckduckgo.com",
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
            "stackoverflow.com",
            "stackexchange.com",
            "askubuntu.com",
            "superuser.com",
            "serverfault.com",
            "www.w3.org",
            "tools.ietf.org",
            "datatracker.ietf.org",
            "www.rfc-editor.org",
            "www.bbc.com",
            "www.bbc.co.uk",
            "www.reuters.com",
            "apnews.com",
            "www.nature.com",
            "www.science.org",
            "data.gov",
            "catalog.data.gov",
            "www.census.gov",
            "www.nasa.gov",
            "data.nasa.gov",
            "www.cdc.gov",
            "www.nih.gov",
            "www.who.int",
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
        _joined(("/etc/pas", "swd")),
        _joined(("/etc/sh", "adow")),
        "/etc/sudoers",
        "/etc/ssh/",
    )
    _SENSITIVE_FILE_RE = re.compile(r"^/proc/(?:self|\d+)/(?:environ|cmdline|task/\d+/environ)$")

    def _normalize_host(host: str) -> str:
        if not host:
            return ""
        h = host.strip().lower().rstrip(".")
        if "\\" in h:
            # urllib3 ends the host at a backslash, httpx reads it as userinfo: trust neither.
            return h
        if "@" in h:
            h = h.rsplit("@", 1)[1]
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

    # The bare method fallback is fuzzy, so it fires only when huggingface_hub is imported.
    _HF_IMPORT_MODULES = (
        "huggingface_hub",
        "hf_api",
        "huggingface_hub.hf_api",
    )

    def _module_has_hf_import(tree: ast.AST) -> bool:
        for n in _tree_nodes(tree):
            if isinstance(n, ast.Import):
                for alias in n.names:
                    if alias.name.split(".", 1)[0] in _HF_IMPORT_MODULES:
                        return True
            elif isinstance(n, ast.ImportFrom):
                root = (n.module or "").split(".", 1)[0]
                if root in _HF_IMPORT_MODULES:
                    return True
            elif isinstance(n, ast.Call) and n.args:
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
        """Return the HF upload method name (`upload_file`, ...) or None. Covers the Attribute and
        bare-Name forms; the bare-name branch fires only when an HF import is in scope so
        paramiko/boto3 don't false-positive."""
        if not _hf_in_scope:
            return None
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in _UPLOAD_HF_METHODS:
            return f.attr
        if isinstance(f, ast.Name) and f.id in _UPLOAD_HF_METHODS:
            return f.id
        return None

    # The sandbox env strips credentials, so any value here is hard-coded or lifted.
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

    # preupload_lfs_files sends file bytes on its own, so it is gated like a commit.
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
        """True if any node in the subtree resolves to an env/process read. Walks the whole subtree
        (not just the root) to catch wrappers like `str(os.environ)`. Covers
        os.environ[/.get]/os.getenv, bare getenv, and subprocess.{run,check_output,...} that
        could lift parent env via printenv."""
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
            return True
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
        """Inspect an HF upload call; return a violation reason or None. Policy: HF uploads are
        allowed only when (a) no sensitive kwarg is set, (b) no positional / keyword value reads
        `os.environ` or related env readers, and (c) the path arg is a sandbox-local literal: a
        relative string with no `..`, an `open(<literal>)`, or inline bytes. Dynamic / variable
        paths are rejected since safety can't be proven statically and a wrong-allow means
        credential exfiltration."""
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
            # A splat can smuggle operations or a token; scan before resolving.
            if any(isinstance(a, ast.Starred) for a in node.args or []):
                return _HF_UPLOAD_PATH_VIOLATION
            if any(kw.arg is None for kw in node.keywords or []):
                return _HF_UPLOAD_PATH_VIOLATION
            operations_node: ast.AST | None = node.args[1] if len(node.args or []) > 1 else None
            for kw in node.keywords or []:
                if kw.arg == ops_kwarg:
                    operations_node = kw.value
                    break
            if operations_node is None:
                return None
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
            # No by-name exemption for delete/copy (names can be rebound); Add checks both positionals and
            # non-literal keywords.
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

    # Past this many candidates the destination is unreadable, so the screen cannot be stalled.
    _LITERAL_CANDIDATE_CAP = 8

    def _stored_names(targets) -> "list[str]":
        """Only the names a target actually binds.

        Walking every `ast.Name` under the target also returns the BASE of an attribute or
        subscript, which is read, not rebound: `r.debug = True` left `r` looking rebound and
        dropped the `r -> requests` alias that the runtime still has, so the call after it went
        unrecognised.
        """
        names: list[str] = []
        stack = [t for t in targets if t is not None]
        while stack:
            node = stack.pop()
            if isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, (ast.Tuple, ast.List)):
                stack.extend(node.elts)
            elif isinstance(node, ast.Starred):
                stack.append(node.value)
        return names

    # Exact-type lookup: runs twice per node and ast nodes are never subclassed here.
    _BINDING_NODE_TYPES = frozenset(
        {
            ast.Assign,
            ast.Delete,
            ast.AnnAssign,
            ast.AugAssign,
            ast.NamedExpr,
            ast.For,
            ast.AsyncFor,
            ast.comprehension,
            ast.withitem,
            ast.ExceptHandler,
            ast.MatchAs,
            ast.MatchStar,
            ast.MatchMapping,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Lambda,
            ast.Import,
            ast.ImportFrom,
        }
    )

    _REBOUND_BY_HANDLER = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)

    # A shadow removes recognition, so only believe one that cannot be skipped (not inside
    # if/try/for/while/with).
    _UNCONDITIONAL_SHADOW_TYPES = frozenset(
        {
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.Delete,
            # Imports shadow too, but never the names they bind to a network module themselves.
            ast.Import,
            ast.ImportFrom,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        }
    )

    def _binding_names(node) -> "list[str]":
        """Every name a node binds, in whatever form: assignment, unpacking, walrus, import, def,
        class, parameter, for target, `as` clause, del. One definition of "this name now means
        something else", used both to invalidate module aliases and to collect literal values."""
        if type(node) not in _BINDING_NODE_TYPES:
            return []
        if isinstance(node, (ast.Assign, ast.Delete)):
            return _stored_names(node.targets)
        if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            return _stored_names([node.target])
        if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            return _stored_names([node.target])
        if isinstance(node, ast.withitem):
            return _stored_names([node.optional_vars])
        if isinstance(node, ast.ExceptHandler):
            return [node.name] if node.name else []
        if isinstance(node, (ast.MatchAs, ast.MatchStar)):
            return [node.name] if node.name else []
        if isinstance(node, ast.MatchMapping):
            return [node.rest] if node.rest else []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            args = getattr(node, "args", None)
            slots = (
                []
                if args is None
                else list(args.posonlyargs)
                + list(args.args)
                + list(args.kwonlyargs)
                + [args.vararg, args.kwarg]
            )
            bound = [a.arg for a in slots if a is not None]
            return ([node.name] if getattr(node, "name", None) else []) + bound
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return [
                (alias.asname or alias.name.split(".")[0])
                for alias in node.names
                if (alias.asname or alias.name) != "*"
            ]
        return []

    _REQUEST_SAFE_BINDINGS = frozenset({"urllib", "urllib.request", "urllib.request.Request"})

    def _scope_bodies(nodes, tree):
        """`(scope id, statement list)` for the module and for every function, lambda and class body.

        A shadow belongs to the scope whose body it sits directly in, so this is what says which
        statements can shadow a name for which calls. The module is scope 0. A lambda has an
        expression rather than a body, so it contributes no statements, only its parameters.
        """
        yield 0, getattr(tree, "body", [])
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                yield id(node), node.body

    def _names_bound_to_something_else(nodes) -> "set[str]":
        """Every name bound ANYWHERE in the tree, in ANY scope, to something that is not part of
        `urllib.request.Request`, plus `"*"` when a star import could have supplied it.

        Unwrapping `urlopen(Request(url))` READS PAST a call, which is the one place where adding
        a candidate makes the screen weaker rather than stronger, so it cannot use the accumulating
        alias maps: a nested `def Request(_): return "https://evil.example/x"` really does decide
        what the call inside that function reaches. Scope is ignored in the strict direction here,
        so a binding anywhere is enough to refuse the unwrap and leave the argument reading as a
        call, which the fail-closed rule then refuses.
        """
        out: set[str] = set()
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in _REQUEST_SAFE_BINDINGS:
                        out.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if alias.name == "*":
                        if module != "urllib.request":
                            out.add("*")
                    elif f"{module}.{alias.name}" not in _REQUEST_SAFE_BINDINGS:
                        out.add(alias.asname or alias.name)
            elif isinstance(node, ast.Attribute) and node.attr == "Request":
                # Assigning to any `.Request` attribute refuses every unwrap.
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    out.add("*")
            elif isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name == "setattr" and len(node.args) >= 2:
                    attr = node.args[1]
                    if not isinstance(attr, ast.Constant) or attr.value == "Request":
                        out.add("*")
            else:
                out.update(_binding_names(node))
        return out

    def _collect_literal_names(nodes) -> "dict[str, frozenset[str] | None]":
        """Name -> every string literal it is bound to anywhere in the tree, or None once any
        binding is something this screen cannot read. Order and scope are ignored on purpose: the
        call site is then checked against EVERY value the name can hold, which stays sound without
        reasoning about which branch ran or which loop iteration this is. Reading only the newest
        binding would allow `if f: url = evil` / `else: url = allowed` / `get(url)`."""
        values: "dict[str, set[str] | None]" = {}

        def bind(name: str, literal: "str | None") -> None:
            current = values[name] if name in values else set()
            if current is None or literal is None:
                values[name] = None
                return
            current.add(literal)
            values[name] = None if len(current) > _LITERAL_CANDIDATE_CAP else current

        for node in nodes:
            literal_targets: list[str] = []
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                value = getattr(node, "value", None)
                literal = value.value if isinstance(value, ast.Constant) else None
                if isinstance(literal, str):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    literal_targets = [t.id for t in targets if isinstance(t, ast.Name)]
            for name in _binding_names(node):
                bind(name, literal if name in literal_targets else None)
        return {k: (None if v is None else frozenset(v)) for k, v in values.items()}

    def _literal_str_prefixes(node, names) -> "list[tuple[str, bool]]":
        """Every text a string expression is statically known to START with, one entry per value its
        names can hold, each with whether that text is the whole value. The head is what decides the
        destination: a URL's scheme and host sit in front of whatever a concatenation or an f-string
        appends at runtime. `("", False)` means unreadable, so a caller can fail closed on it."""
        if isinstance(node, ast.Constant):
            return [(node.value, True)] if isinstance(node.value, str) else [("", False)]
        if isinstance(node, ast.Name):
            bound = names.get(node.id)
            return [(v, True) for v in sorted(bound)] if bound else [("", False)]
        if isinstance(node, ast.NamedExpr):
            return _literal_str_prefixes(node.value, names)
        if isinstance(node, ast.JoinedStr):
            text = ""
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    text += part.value
                    continue
                return [(text, False)]
            return [(text, True)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            out: "list[tuple[str, bool]]" = []
            for left, left_whole in _literal_str_prefixes(node.left, names):
                if not left_whole:
                    out.append((left, False))
                    continue
                out.extend(
                    (left + right, right_whole)
                    for right, right_whole in _literal_str_prefixes(node.right, names)
                )
                if len(out) > _LITERAL_CANDIDATE_CAP:
                    return [("", False)]
            return out or [("", False)]
        return [("", False)]

    def _alternatives(value) -> "list[ast.AST]":
        """Every expression a value can evaluate to (conditional, `or`, walrus, `await`)."""
        out: "list[ast.AST]" = []
        stack = [value]
        while stack:
            cur = stack.pop()
            if isinstance(cur, ast.IfExp):
                stack.extend((cur.body, cur.orelse))
            elif isinstance(cur, ast.BoolOp):
                stack.extend(cur.values)
            elif isinstance(cur, (ast.NamedExpr, ast.Await)):
                stack.append(cur.value)
            else:
                out.append(cur)
        return out

    def _constant_getattr(node) -> "ast.Attribute | None":
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            return ast.Attribute(value = node.args[0], attr = node.args[1].value, ctx = ast.Load())
        return None

    def _reexported(fq: str) -> "str | None":
        """The listed top-level re-export of a submodule name: `httpx._api.get` is `httpx.get`."""
        parts = fq.split(".")
        if len(parts) < 3 or parts[0] not in _NETWORK_ROOTS:
            return None
        if not all(p[:1].islower() or p[:1] == "_" for p in parts[1:-1]):
            return None
        top = f"{parts[0]}.{parts[-1]}"
        return top if top in _NETWORK_DESTINATION_ARG or top in _CLIENT_CLASSES else None

    def _returns_of(function_name: str) -> str:
        return f"{function_name}()"

    # Only these read proxy env vars (aiohttp only with trust_env).
    _ENV_PROXY_CLIENTS = ("requests.", "urllib.request.")

    def _reads_proxy_environment(node: ast.Call, recognised) -> bool:
        trust = next((kw.value for kw in node.keywords if kw.arg == "trust_env"), None)
        disabled = isinstance(trust, ast.Constant) and trust.value is False
        if any(c.startswith("httpx.") for c in recognised):
            return not disabled
        if any(c.startswith(_ENV_PROXY_CLIENTS) for c in recognised):
            return True
        if any(c.startswith("aiohttp.") for c in recognised):
            return trust is not None and not disabled
        return False

    _ENV_PROXY_VARIABLES = frozenset(
        {"http_proxy", "https_proxy", "all_proxy", "ws_proxy", "wss_proxy", "ftp_proxy"}
    )

    _ROUTE_KEYWORDS = {
        "paramiko.SSHClient.": ("sock",),
        "paramiko.client.SSHClient.": ("sock",),
        "fabric.": ("gateway",),
        "asyncssh.": ("tunnel", "proxy_command"),
    }
    _ROUTE_POSITIONS = {
        "paramiko.SSHClient.connect": {10: "sock"},
        "paramiko.client.SSHClient.connect": {10: "sock"},
        "fabric.Connection": {4: "gateway", 7: "connect_kwargs"},
        "fabric.connection.Connection": {4: "gateway", 7: "connect_kwargs"},
    }
    _UNREADABLE = ast.Name(id = "<unreadable>", ctx = ast.Load())

    def _is_no_proxy(key) -> bool:
        return isinstance(key, ast.Constant) and key.value == "no_proxy"

    def _paired(target, value):
        if not isinstance(target, (ast.Tuple, ast.List)):
            yield target, value
            return
        if not isinstance(value, (ast.Tuple, ast.List)) or any(
            isinstance(e, ast.Starred) for e in value.elts
        ):
            return
        elts = target.elts
        star = next((i for i, e in enumerate(elts) if isinstance(e, ast.Starred)), None)
        if star is None:
            if len(elts) != len(value.elts):
                return
            pairs = list(zip(elts, value.elts))
        else:
            before, after = elts[:star], elts[star + 1 :]
            if len(value.elts) < len(before) + len(after):
                return
            pairs = list(zip(before, value.elts)) + list(
                zip(after, value.elts[len(value.elts) - len(after) :])
            )
        for t, v in pairs:
            yield from _paired(t, v)

    class NetworkAndIoVisitor(ast.NodeVisitor):
        def __init__(self):
            # Accumulated, not overwritten: the map is not scope aware, so resolution must be monotone.
            self.module_aliases: dict[str, set[str]] = {}
            # Accumulated for the same reason.
            self.func_aliases: dict[str, set[str]] = {}
            # Built only when the source can name a network module; ordinary code skips these walks.
            nodes = _tree_nodes(tree) if network_possible else ()
            self.literal_names = _collect_literal_names(nodes) if network_possible else {}
            self.rebound_anywhere = (
                _names_bound_to_something_else(nodes) if network_possible else frozenset()
            )
            # Unconditional module-level statements that may shadow, mapped to their enclosing scope.
            self.unconditional_shadows: "dict[int, int]" = {}
            if network_possible:
                for scope, body in _scope_bodies(nodes, tree):
                    for stmt in body:
                        if type(stmt) in _UNCONDITIONAL_SHADOW_TYPES:
                            self.unconditional_shadows[id(stmt)] = scope
                        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.NamedExpr):
                            # A statement-level walrus certainly runs; one buried in an expression may not.
                            self.unconditional_shadows[id(stmt.value)] = scope
            self.scope_stack: list[int] = [0]
            # Star imports bind names this file cannot enumerate; resolve callees against them.
            self.star_modules: set[str] = set()
            # A shadow only affects calls that cannot run before it.
            self.shadow_lines: "dict[tuple[str, int], list[tuple[int, int]]]" = {}
            self.star_lines: "list[tuple[int, int]]" = []
            # A shadow counts only while no later alias registration follows it.
            self.alias_lines: "dict[str, list[tuple[int, int]]]" = {}
            # Two passes: aliases first, then calls, since function bodies run after the module is read.
            self.collecting = True
            self.aliases_possible = network_possible
            self.instance_aliases: "dict[str, set[str]]" = {}
            self.receiver_destinations: "dict[str, list[tuple[str, ast.AST]]]" = {}
            self.path_links: "dict[str, set[str]]" = {}
            self.proxy_owners: "dict[str, set[tuple[str, str]]]" = {}
            self.class_family: "dict[int, str]" = {}
            self.class_names: "dict[str, str]" = {}
            self.properties: "list[tuple[str, str]]" = []
            self.class_inits: "dict[str, list[ast.AST]]" = {}
            self.method_self: "dict[int, tuple[str, str]]" = {}
            if network_possible:
                classes = [n for n in nodes if isinstance(n, ast.ClassDef)]
                parent = {c.name: c.name for c in classes}

                def find(name):
                    while parent[name] != name:
                        name = parent[name]
                    return name

                for c in classes:
                    for base in c.bases:
                        if isinstance(base, ast.Name) and base.id in parent:
                            parent[find(base.id)] = find(c.name)
                for c in classes:
                    family = f"<{find(c.name)}>"
                    self.class_family[id(c)] = family
                    self.class_names[c.name] = family
                    for fn in c.body:
                        if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                            self.class_inits.setdefault(c.name, []).append(fn)
                    for fn in c.body:
                        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and not any(
                            (getattr(d, "id", None) or getattr(d, "attr", None)) == "staticmethod"
                            for d in fn.decorator_list
                        ):
                            params = fn.args.posonlyargs + fn.args.args
                            if params:
                                self.method_self[id(fn)] = (params[0].arg, family)
                            if any(getattr(d, "id", None) == "property" for d in fn.decorator_list):
                                self.properties.append((family, fn.name))
                bases = {
                    c.name: [b.id for b in c.bases if isinstance(b, ast.Name)] for c in classes
                }
                for name in bases:
                    seen, stack = {name}, [name]
                    while stack and name not in self.class_inits:
                        for base in bases.get(stack.pop(), ()):
                            if base in self.class_inits:
                                self.class_inits[name] = self.class_inits[base]
                                break
                            if base not in seen:
                                seen.add(base)
                                stack.append(base)
            self.self_names: "list[tuple[str, str]]" = []
            self.local_functions: "dict[str, list[ast.AST]]" = {}
            self.callable_aliases: "dict[str, set[str]]" = {}
            self.def_names: "dict[int, str]" = {}
            self.local_methods: "dict[str, list[tuple[ast.AST, int]]]" = {}
            if network_possible:
                for fn in nodes:
                    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.local_functions.setdefault(fn.name, []).append(fn)
                        self.def_names[id(fn)] = fn.name
                    elif isinstance(fn, (ast.Assign, ast.AnnAssign)) and isinstance(
                        fn.value, ast.Lambda
                    ):
                        for target in getattr(fn, "targets", None) or [fn.target]:
                            if isinstance(target, ast.Name):
                                self.local_functions.setdefault(target.id, []).append(fn.value)
                    elif isinstance(fn, ast.ClassDef):
                        for m in fn.body:
                            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                skip = 1 if id(m) in self.method_self else 0
                                self.local_methods.setdefault(m.name, []).append((m, skip))
            # Re-read after gathering, when a path they read gains a client.
            self.flows: "list[tuple]" = []
            self.flows_from: "dict[str, list[int]]" = {}
            self.pending_calls: "list[tuple]" = []
            self.pending_proxies: "list[tuple]" = []
            self.env_proxies: "list[ast.AST]" = []
            self.dict_literals: "dict[str, list[ast.Dict]]" = {}
            self.dict_entries: "dict[str, list[tuple]]" = {}
            self.dict_mutated: "set[str]" = set()
            self.os_names: "set[str]" = {"os"}
            self.environ_names: "set[str]" = set()
            for imp in nodes:
                if isinstance(imp, ast.Import):
                    for alias in imp.names:
                        if alias.name == "os":
                            self.os_names.add(alias.asname or "os")
                elif isinstance(imp, ast.ImportFrom) and imp.module == "os":
                    for alias in imp.names:
                        if alias.name in ("environ", "environb"):
                            self.environ_names.add(alias.asname or alias.name)

        def _instance_key(self, target) -> "str | None":
            family = self.class_family.get(self.scope_stack[-1])
            if family is not None and isinstance(target, ast.Name):
                return f"{family}.{target.id}"
            return self._receiver_path(target)

        def _record_flow(self, target, value, at, node) -> None:
            if not self.collecting or self._instance_key(target) is None:
                return
            self.flows.append(
                (target, value, at, node, tuple(self.scope_stack), tuple(self.self_names))
            )

        def _index_flows(self) -> None:
            """Runs after the gathering pass, so an alias assigned below its use is already known."""
            for index, (_t, value, _at, _n, scopes, selves) in enumerate(self.flows):
                self.scope_stack, self.self_names = list(scopes), list(selves)
                for alt in _alternatives(value):
                    if isinstance(alt, ast.Call):
                        for callee in self._local_callees(alt.func):
                            self.flows_from.setdefault(_returns_of(callee), []).append(index)
                    else:
                        path = self._receiver_path(alt)
                        if path is not None:
                            self.flows_from.setdefault(path, []).append(index)

        def _local_callees(self, func) -> "set[str]":
            if isinstance(func, ast.Name):
                names, seen, stack = set(), {func.id}, [func.id]
                while stack:
                    name = stack.pop()
                    if name in self.local_functions:
                        names.add(name)
                    for source in self.callable_aliases.get(name, ()):
                        if source not in seen:
                            seen.add(source)
                            stack.append(source)
                return names
            if isinstance(func, ast.Attribute) and func.attr in self.local_methods:
                return {func.attr}
            return set()

        def _bind_call_arguments(self, node) -> None:
            if isinstance(node.func, ast.Name):
                targets = [
                    (fn, skip)
                    for name in sorted(self._local_callees(node.func))
                    for fn in self.local_functions.get(name, ())
                    for skip in ((0, 1) if id(fn) in self.method_self else (0,))
                ] + [(fn, 1) for fn in self.class_inits.get(node.func.id, ())]
            elif isinstance(node.func, ast.Attribute):
                methods = self.local_methods.get(node.func.attr, [])
                targets = methods + [(fn, 0) for fn, skip in methods if skip]
            else:
                return
            at = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
            for fn, skip in targets:
                positional = (fn.args.posonlyargs + fn.args.args)[skip:]
                pairs = []
                for param, arg in zip(positional, node.args):
                    if isinstance(arg, ast.Starred):
                        break
                    pairs.append((param.arg, arg))
                named = {a.arg for a in positional + fn.args.kwonlyargs}
                pairs += [(kw.arg, kw.value) for kw in node.keywords if kw.arg in named]
                for param, arg in pairs:
                    target = ast.Name(id = param, ctx = ast.Store())
                    self._link(target, arg)
                    if self._is_environ(arg):
                        self.environ_names.add(param)
                    if isinstance(arg, ast.Name) and arg.id in self.os_names:
                        self.os_names.add(param)
                    if isinstance(arg, ast.Attribute) and arg.attr in _DESTINATION_ATTRS:
                        owner = self._receiver_path(arg.value)
                        if owner is not None:
                            self.proxy_owners.setdefault(param, set()).add((owner, arg.attr))
                    self._record_flow(target, arg, at, fn)

        def resolve_flows(self) -> None:
            """Fixpoint: each pass past the subset check adds a client to a path, so this terminates."""
            for call, scopes, selves in self.pending_calls:
                self.scope_stack, self.self_names = list(scopes), list(selves)
                self._bind_call_arguments(call)
            for target, value, mutated, scopes, selves in self.pending_proxies:
                self.scope_stack, self.self_names = list(scopes), list(selves)
                self._apply_proxy(target, value, mutated)
            for name in self.environ_names:
                for key, value in self.dict_entries.get(name, ()):
                    self._record_env_proxy(key, value)
                if name in self.dict_mutated:
                    self.env_proxies.append(_UNREADABLE)
            for family, name in self.properties:
                attribute = ast.Name(id = f"{family}.{name}", ctx = ast.Store())
                returned = ast.Name(id = _returns_of(name), ctx = ast.Load())
                self.flows.append((attribute, returned, (0, 0), attribute, (0,), ()))
            self._index_flows()
            queue = list(range(len(self.flows)))
            while queue:
                target, value, at, node, scopes, selves = self.flows[queue.pop()]
                self.scope_stack, self.self_names = list(scopes), list(selves)
                found = self._instances_named_by(value, at)
                key = self._instance_key(target)
                if key is None or found <= self.instance_aliases.get(key, set()):
                    continue
                self._register(target, (set(), set(), found), node)
                queue.extend(self.flows_from.get(key, ()))
            self.scope_stack, self.self_names = [0], []

        def _receiver_path(self, expr) -> "str | None":
            parts: list[str] = []
            cur = expr
            while isinstance(cur, ast.Attribute):
                parts.insert(0, cur.attr)
                cur = cur.value
            if not isinstance(cur, ast.Name):
                return None
            root = next((f for name, f in reversed(self.self_names) if name == cur.id), cur.id)
            return ".".join([root] + parts)

        def _roots(self, name: str) -> "list[str]":
            roots = [next((f for n, f in reversed(self.self_names) if n == name), name)]
            if name in self.class_names:
                roots.append(self.class_names[name])
            family = self.class_family.get(self.scope_stack[-1])
            if family is not None:
                roots.append(f"{family}.{name}")
            for root in list(roots):
                roots += [f for f in self.instance_aliases.get(root, ()) if f.startswith("<")]
            return list(dict.fromkeys(roots))

        def _path_variants(self, expr) -> "list[str]":
            parts: list[str] = []
            cur = expr
            while isinstance(cur, ast.Attribute):
                parts.insert(0, cur.attr)
                cur = cur.value
            if not isinstance(cur, ast.Name):
                return []
            return [".".join([root] + parts) for root in self._roots(cur.id)]

        def _instances_named_by(self, value, at) -> "set[str]":
            found: set[str] = set()
            for alt in _alternatives(value):
                if isinstance(alt, ast.Call):
                    found.update(
                        c for c in self._fq_candidates(alt.func, at) if c in _CLIENT_CLASSES
                    )
                    for callee in self._local_callees(alt.func):
                        found.update(self.instance_aliases.get(_returns_of(callee), ()))
                    if isinstance(alt.func, ast.Name) and alt.func.id in self.class_names:
                        found.add(self.class_names[alt.func.id])
                    if (
                        isinstance(alt.func, ast.Name)
                        and alt.func.id == "super"
                        and not alt.args
                        and self.self_names
                    ):
                        found.add(self.self_names[-1][1])
                    continue
                receiver = isinstance(alt, ast.Name) and any(
                    n == alt.id for n, _f in self.self_names
                )
                if isinstance(alt, ast.Name) and not receiver and self._is_shadowed(alt.id, at):
                    continue
                for path in self._path_variants(alt):
                    found.update(self.instance_aliases.get(path, ()))
            return found

        def _shadowing_names(self, node) -> "list[str]":
            """The names a node binds IN THE ENCLOSING scope, which is the only scope that can
            shadow an imported name. A def or class binds its own name there and its parameters
            inside itself, so only the name counts."""
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return [node.name]
            if isinstance(node, ast.Lambda):
                return []
            return _binding_names(node)

        def _rebind(
            self,
            node,
            exempt = (),
        ) -> None:
            if self.collecting and id(node) in self.unconditional_shadows:
                scope = self.unconditional_shadows[id(node)]
                # Position, not just line: all three can sit on one line.
                where = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # A def/class binds its name after defaults and bases run.
                    where = (node.body[0].lineno, node.body[0].col_offset)
                for name in self._shadowing_names(node):
                    if name in exempt:
                        # A statement that registers an alias must not also shadow the name it bound.
                        continue
                    # The module set is not dropped (see __init__); function aliases are.
                    self.shadow_lines.setdefault((name, scope), []).append(where)

        def generic_visit(self, node):
            """`ast.NodeVisitor.generic_visit`, inlined, plus the rebinding hook.

            Wrapping it instead would spend a third Python frame per level of nesting and so cut
            the nesting this screen survives by a third: measured, a 494-term `+` chain analysed
            and a 329-term one raised RecursionError, where the limit is 494 either side now. The
            handlers that record an alias do their own rebinding FIRST, so they are skipped here:
            they call this after registering, and a second sweep would pop what they just set.
            """
            if self.unconditional_shadows and not isinstance(node, _REBOUND_BY_HANDLER):
                self._rebind(node)
            for _field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            self.visit(item)
                elif isinstance(value, ast.AST):
                    self.visit(value)

        def _register_alias(self, name: str, node) -> None:
            """Note where a network alias was bound to `name`, so a shadow older than this one no
            longer applies. See `_is_shadowed`."""
            self.alias_lines.setdefault(name, []).append(
                (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
            )

        def _is_shadowed(
            self,
            name: str,
            at,
            after_star: bool = False,
        ) -> bool:
            """Whether a module-level rebinding of `name` has certainly happened by `at`.

            Positions are `(lineno, col_offset)`, so a semicolon-separated
            `from requests import get as fetch; fetch = print; fetch(url)` orders correctly where
            comparing lines alone made the rebinding invisible.

            A call inside a function, lambda or class body is never shadowed: the body can be
            invoked at any point, including before the rebinding. At module level the rebinding
            counts only for what comes AFTER it, and only while no alias registration comes between
            the two: the last `import`, `from ... import` or module-carrying assignment before the
            call supersedes every shadow older than it. For a star-imported name the star import is
            that registration, since it rebinds every exported name.
            """
            registrations = self.star_lines if after_star else self.alias_lines.get(name, ())
            floor = max((where for where in registrations if where < at), default = (0, -1))
            return any(
                floor < where < at
                for where in self.shadow_lines.get((name, self.scope_stack[-1]), ())
            )

        def _star_imported_fq(self, name: str, at) -> "str | None":
            if self._is_shadowed(name, at, after_star = True):
                return None
            for module in sorted(self.star_modules):
                fq = f"{module}.{name}"
                if any(fq.startswith(p) for p in _NETWORK_FQ_PREFIXES):
                    return fq
            return None

        def _visit_scope(self, node):
            self._rebind(node)
            if self.collecting:
                where = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
                for name in _binding_names(node):
                    if name != getattr(node, "name", None):
                        self.shadow_lines.setdefault((name, id(node)), []).append(where)
            # Decorators, defaults, annotations and bases run in the enclosing scope; only the body is
            # the new scope.
            for field, value in ast.iter_fields(node):
                if field == "body":
                    continue
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            self.visit(item)
                elif isinstance(value, ast.AST):
                    self.visit(value)
            if self.collecting and isinstance(node, ast.ClassDef):
                where = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
                inherited = {
                    c
                    for base in node.bases
                    for c in self._fq_candidates(base, where)
                    if c in _CLIENT_CLASSES or c in _NETWORK_DESTINATION_ARG
                }
                if inherited:
                    self.func_aliases.setdefault(node.name, set()).update(inherited)
                    family = self.class_family.get(id(node))
                    if family is not None:
                        self.instance_aliases.setdefault(family, set()).update(inherited)
                    self._register_alias(node.name, node.body[0])
            args = getattr(node, "args", None)
            if self.collecting and args is not None:
                positional = args.posonlyargs + args.args
                defaults = list(
                    zip(positional[len(positional) - len(args.defaults) :], args.defaults)
                )
                defaults += [
                    (a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None
                ]
                where = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
                for param, default in defaults:
                    self._carry(ast.Name(id = param.arg, ctx = ast.Store()), default, where, node)
            self.scope_stack.append(id(node))
            self_name = self.method_self.get(id(node))
            if self_name is not None:
                self.self_names.append(self_name)
            try:
                body = node.body
                if isinstance(body, list):
                    for item in body:
                        if isinstance(item, ast.AST):
                            self.visit(item)
                elif isinstance(body, ast.AST):
                    self.visit(body)
            finally:
                self.scope_stack.pop()
                if self_name is not None:
                    self.self_names.pop()

        visit_FunctionDef = _visit_scope
        visit_AsyncFunctionDef = _visit_scope
        visit_ClassDef = _visit_scope
        visit_Lambda = _visit_scope

        def visit_Import(self, node):
            if not self.collecting:
                self.generic_visit(node)
                return
            registered: set[str] = set()
            for alias in node.names:
                if alias.asname and (
                    alias.name in _NETWORK_MODULES or alias.name.partition(".")[0] in _NETWORK_ROOTS
                ):
                    self.module_aliases.setdefault(alias.asname, set()).add(alias.name)
                    self._register_alias(alias.asname, node)
                    registered.add(alias.asname)
                elif not alias.asname and alias.name.partition(".")[0] in _NETWORK_ROOTS:
                    # `import requests` still binds the name; `import urllib.request` binds `urllib`.
                    self._register_alias(alias.name.partition(".")[0], node)
                    registered.add(alias.name.partition(".")[0])
            self._rebind(node, exempt = registered)
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if not self.collecting:
                self.generic_visit(node)
                return
            registered: set[str] = set()
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    if module in _NETWORK_MODULES:
                        self.star_modules.add(module)
                        self.star_lines.append(
                            (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
                        )
                        # A star import rebinds every exported name, so earlier shadows are dropped.
                    continue
                bound = alias.asname or alias.name
                fq = f"{module}.{alias.name}"
                if fq in _NETWORK_MODULES:
                    self.module_aliases.setdefault(bound, set()).add(fq)
                    self._register_alias(bound, node)
                    registered.add(bound)
                elif module in _NETWORK_MODULES or _reexported(fq):
                    self.func_aliases.setdefault(bound, set()).add(fq)
                    self._register_alias(bound, node)
                    registered.add(bound)
            self._rebind(node, exempt = registered)
            self.generic_visit(node)

        def _modules_named_by(self, value, at) -> "set[str]":
            """Every network module a value can name, following aliases, so `r = requests` keeps
            `r.get(...)` screened instead of letting the assignment shed the module.

            EVERY candidate is carried, not the first: `import requests as r` with a nested,
            never-executed `import aiohttp as r` leaves `r` as `requests` at runtime, and
            collapsing the set to one candidate resolved the later `s = r; s.get(...)` to the
            unrecognised `aiohttp.get` and let a hostile host through.
            """
            alts = _alternatives(value)
            if alts != [value]:
                return set().union(*(self._modules_named_by(alt, at) for alt in alts))
            parts: list[str] = []
            cur = value
            while isinstance(cur, ast.Attribute):
                parts.insert(0, cur.attr)
                cur = cur.value
            if not isinstance(cur, ast.Name):
                return set()
            parts.insert(0, cur.id)
            if self._is_shadowed(parts[0], at):
                # The copy's source was rebound earlier, so it no longer names the module.
                return set()
            heads = self.module_aliases.get(parts[0]) or {parts[0]}
            found = {".".join(head.split(".") + parts[1:]) for head in heads}
            # A package containing a network module counts too (`u = urllib`).
            return {
                fq
                for fq in found
                if fq in _NETWORK_MODULES
                or any(module.startswith(f"{fq}.") for module in _NETWORK_MODULES)
            }

        def _functions_named_by(self, value, at) -> "set[str]":
            """Every network FUNCTION a value can name, the counterpart of `_modules_named_by`.

            Without it an assignment shed the function the way it once shed the module:
            `from requests import get as fetch` then `fetch = fetch`, or `g = fetch`, recorded the
            target as shadowed and left the later call with no candidate at all.
            """
            alts = _alternatives(value)
            if alts != [value]:
                return set().union(*(self._functions_named_by(alt, at) for alt in alts))
            value = _constant_getattr(value) or value
            if isinstance(value, ast.Name):
                if self._is_shadowed(value.id, at):
                    return set()
                return set(self.func_aliases.get(value.id) or ())
            if isinstance(value, ast.Attribute):
                return {
                    fq
                    for fq in self._fq_candidates(value, at)
                    if any(fq.startswith(prefix) for prefix in _NETWORK_FQ_PREFIXES)
                }
            return set()

        def _named_by(self, value, at) -> "tuple[set[str], set[str], set[str]]":
            return (
                self._modules_named_by(value, at),
                self._functions_named_by(value, at),
                self._instances_named_by(value, at),
            )

        def _register(self, target, named, node) -> bool:
            """True when a name took an alias, exempting it from the shadow this statement records."""
            modules, functions, instances = named
            path = self._receiver_path(target)
            family = self.class_family.get(self.scope_stack[-1])
            if family is not None and isinstance(target, ast.Name):
                path = f"{family}.{target.id}"
            if path is not None and instances:
                self.instance_aliases.setdefault(path, set()).update(instances)
            if not isinstance(target, ast.Name):
                return False
            if modules:
                self.module_aliases.setdefault(target.id, set()).update(modules)
            if functions:
                self.func_aliases.setdefault(target.id, set()).update(functions)
            if modules or functions or any(not c.startswith("<") for c in instances):
                self._register_alias(target.id, node)
                return True
            return False

        def _carry(self, target, value, at, node) -> bool:
            self._link(target, value)
            self._record_flow(target, value, at, node)
            return self._register(target, self._named_by(value, at), node)

        def _link(self, target, value) -> None:
            path = self._receiver_path(target)
            if path is None:
                return
            for alt in _alternatives(value):
                other = self._receiver_path(alt)
                if other is not None and other != path:
                    self.path_links.setdefault(path, set()).add(other)
                    self.path_links.setdefault(other, set()).add(path)

        def _linked_paths(self, path: str) -> "set[str]":
            seen = {path}
            stack = [path]
            while stack:
                for other in self.path_links.get(stack.pop(), ()):
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
            return seen

        def _record_proxy(
            self,
            target,
            value,
            mutated = False,
        ) -> None:
            """Applied after call arguments are bound, so `configure(s.proxies)` still reaches `s`."""
            if isinstance(target, ast.Subscript) and self._is_environ(target.value):
                self._record_env_proxy(target.slice, value)
                return
            if isinstance(target, ast.Attribute) and self._is_environ(target):
                self._record_env_mapping(value)
                return
            self.pending_proxies.append(
                (target, value, mutated, tuple(self.scope_stack), tuple(self.self_names))
            )

        def _record_dict_mutation(self, name: str, method: str, args, keywords) -> None:
            if method in ("setdefault", "__setitem__") and len(args) >= 2:
                self.dict_entries.setdefault(name, []).append((args[0], args[1]))
            elif method in ("update", "__ior__"):
                for arg in args:
                    if isinstance(arg, ast.Dict) and None not in arg.keys:
                        self.dict_entries.setdefault(name, []).extend(zip(arg.keys, arg.values))
                    else:
                        self.dict_mutated.add(name)
                for kw in keywords:
                    if kw.arg is None:
                        self.dict_mutated.add(name)
                    else:
                        self.dict_entries.setdefault(name, []).append(
                            (ast.Constant(value = kw.arg), kw.value)
                        )

        def _record_env_mapping(self, mapping) -> None:
            """Mapping merged into the environment; anything unreadable is recorded as unreadable."""
            if isinstance(mapping, ast.Dict):
                for k, v in zip(mapping.keys, mapping.values):
                    if k is None:
                        self.env_proxies.append(_UNREADABLE)
                    else:
                        self._record_env_proxy(k, v)
            elif isinstance(mapping, ast.Name) and mapping.id in self.dict_literals:
                if mapping.id in self.dict_mutated:
                    self.env_proxies.append(_UNREADABLE)
                for literal in self.dict_literals[mapping.id]:
                    self._record_env_mapping(literal)
                for key, value in self.dict_entries.get(mapping.id, ()):
                    self._record_env_proxy(key, value)
            elif (
                isinstance(mapping, ast.Call)
                and isinstance(mapping.func, ast.Name)
                and mapping.func.id == "dict"
                and not mapping.args
            ):
                for kw in mapping.keywords:
                    if kw.arg is None:
                        self.env_proxies.append(_UNREADABLE)
                    else:
                        self._record_env_proxy(ast.Constant(value = kw.arg), kw.value)
            elif isinstance(mapping, (ast.List, ast.Tuple)) and all(
                isinstance(e, ast.Tuple) and len(e.elts) == 2 for e in mapping.elts
            ):
                for e in mapping.elts:
                    self._record_env_proxy(e.elts[0], e.elts[1])
            else:
                self.env_proxies.append(_UNREADABLE)

        def _is_environ(self, node) -> bool:
            node = _constant_getattr(node) or node
            if isinstance(node, ast.Name):
                return node.id in self.environ_names
            return (
                isinstance(node, ast.Attribute)
                and node.attr in ("environ", "environb")
                and isinstance(node.value, ast.Name)
                and node.value.id in self.os_names
            )

        def _record_env_proxy(self, key, value) -> None:
            """requests, httpx and urllib honour the proxy environment by default."""
            if isinstance(key, ast.Constant):
                names = [key.value] if isinstance(key.value, str) else []
                if isinstance(key.value, bytes):
                    names = [key.value.decode("latin-1")]
            elif isinstance(key, ast.Name) and self.literal_names.get(key.id):
                names = list(self.literal_names[key.id])
            else:
                self.env_proxies.append(_UNREADABLE)
                return
            if any(name.lower() in _ENV_PROXY_VARIABLES for name in names):
                self.env_proxies.append(value)

        def _apply_proxy(self, target, value, mutated) -> None:
            if isinstance(target, ast.Subscript) and self._is_environ(target.value):
                self._record_env_proxy(target.slice, value)
                return
            if isinstance(target, ast.Subscript):
                if _is_no_proxy(target.slice):
                    return
                target, mutated = target.value, True
            if isinstance(target, ast.Attribute) and target.attr in _DESTINATION_ATTRS:
                owners = {(self._receiver_path(target.value), target.attr)}
            elif mutated and self._receiver_path(target) is not None:
                owners = set().union(
                    *(
                        self.proxy_owners.get(path, ())
                        for path in self._linked_paths(self._receiver_path(target))
                    )
                )
            else:
                return
            for owner, attr in owners:
                if owner is not None:
                    self.receiver_destinations.setdefault(owner, []).append((attr, value))

        def visit_Assign(self, node):
            self._visit_binding(node, node.targets, node.value)

        def visit_AnnAssign(self, node):
            if node.value is None:
                if self.collecting:
                    self._rebind(node)
                self.generic_visit(node)
                return
            self._visit_binding(node, [node.target], node.value)

        def _visit_binding(self, node, targets, value_node):
            if not self.collecting:
                self.generic_visit(node)
                return
            at = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
            pairs = [
                (target, value, self._named_by(value, at))
                for whole in targets
                for target, value in _paired(whole, value_node)
            ]
            if not isinstance(value_node, (ast.Tuple, ast.List)):
                for whole in targets:
                    if isinstance(whole, (ast.Tuple, ast.List)):
                        for elt in whole.elts:
                            elt = elt.value if isinstance(elt, ast.Starred) else elt
                            self._record_flow(elt, value_node, at, node)
            registered: set[str] = set()
            for target, value, named in pairs:
                self._record_proxy(target, value)
                self._link(target, value)
                self._record_flow(target, value, at, node)
                if isinstance(target, ast.Name) and isinstance(value, ast.Dict):
                    self.dict_literals.setdefault(target.id, []).append(value)
                if isinstance(target, ast.Name) and self._is_environ(value):
                    self.environ_names.add(target.id)
                if (
                    isinstance(target, ast.Name)
                    and isinstance(value, ast.Name)
                    and value.id in self.os_names
                ):
                    self.os_names.add(target.id)
                if isinstance(target, ast.Name) and isinstance(value, ast.Lambda):
                    returns = ast.Name(id = _returns_of(target.id), ctx = ast.Store())
                    self._record_flow(returns, value.body, at, node)
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    self._record_dict_mutation(
                        target.value.id, "__setitem__", [target.slice, value], []
                    )
                if isinstance(target, ast.Name):
                    for alt in _alternatives(value):
                        if isinstance(alt, ast.Name) and alt.id != target.id:
                            self.callable_aliases.setdefault(target.id, set()).add(alt.id)
                        elif isinstance(alt, ast.Attribute) and alt.attr in self.local_methods:
                            self.callable_aliases.setdefault(target.id, set()).add(alt.attr)
                if isinstance(value, ast.Attribute) and value.attr in _DESTINATION_ATTRS:
                    owner = self._receiver_path(value.value)
                    alias = self._receiver_path(target)
                    if owner is not None and alias is not None:
                        self.proxy_owners.setdefault(alias, set()).add((owner, value.attr))
                if self._register(target, named, node):
                    registered.add(target.id)
            self._rebind(node, exempt = registered)
            self.generic_visit(node)

        def visit_AugAssign(self, node):
            if self.collecting:
                if self._is_environ(node.target):
                    self._record_env_mapping(node.value)
                else:
                    if isinstance(node.target, ast.Name):
                        self._record_dict_mutation(node.target.id, "__ior__", [node.value], [])
                    self._record_proxy(node.target, node.value, mutated = True)
            self.generic_visit(node)

        def visit_NamedExpr(self, node):
            if self.collecting and isinstance(node.target, ast.Name):
                at = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
                self._carry(node.target, node.value, at, node)
            self.generic_visit(node)

        def visit_Return(self, node):
            function = self.def_names.get(self.scope_stack[-1])
            if self.collecting and node.value is not None and function is not None:
                at = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
                returns = ast.Name(id = _returns_of(function), ctx = ast.Store())
                self._record_flow(returns, node.value, at, node)
                if isinstance(node.value, (ast.Tuple, ast.List)):
                    for elt in node.value.elts:
                        self._record_flow(returns, elt, at, node)
            self.generic_visit(node)

        def _carry_loop(self, node) -> None:
            if self.collecting and isinstance(node.iter, (ast.List, ast.Tuple, ast.Set)):
                for elt in node.iter.elts:
                    if isinstance(elt, ast.Starred):
                        continue
                    at = (getattr(elt, "lineno", 0), getattr(elt, "col_offset", 0))
                    for target, value in _paired(node.target, elt):
                        self._carry(target, value, at, elt)

        def visit_For(self, node):
            self._carry_loop(node)
            self.generic_visit(node)

        visit_AsyncFor = visit_For

        def visit_comprehension(self, node):
            self._carry_loop(node)
            self.generic_visit(node)

        def visit_withitem(self, node):
            if self.collecting and node.optional_vars is not None:
                ctx = node.context_expr
                at = (getattr(ctx, "lineno", 0), getattr(ctx, "col_offset", 0))
                self._carry(node.optional_vars, ctx, at, ctx)
            self.generic_visit(node)

        def _fq_candidates(
            self,
            func,
            at = (0, 0),
        ) -> "list[str]":
            """Every fully qualified name a callee could be, written spelling first."""
            alts = _alternatives(func)
            if alts != [func]:
                out: list[str] = []
                for alt in alts:
                    out.extend(c for c in self._fq_candidates(alt, at) if c not in out)
                return out
            parts: list[str] = []
            cur = func
            while True:
                if isinstance(cur, ast.Attribute):
                    parts.insert(0, cur.attr)
                    cur = cur.value
                elif _constant_getattr(cur) is not None:
                    cur = _constant_getattr(cur)
                else:
                    break
            if isinstance(cur, ast.Name):
                parts.insert(0, cur.id)
            written = ".".join(parts) if parts else ""
            candidates = [written] if written else []
            if not self.aliases_possible:
                return candidates
            if len(parts) > 1 and parts[0] in self.module_aliases:
                # Module aliases are shadow-filtered like function aliases.
                if not self._is_shadowed(parts[0], at):
                    for module in sorted(self.module_aliases[parts[0]]):
                        fq = ".".join(module.split(".") + parts[1:])
                        if module not in _NETWORK_MODULES and not any(
                            m.startswith(f"{module}.") for m in _NETWORK_MODULES
                        ):
                            fq = _reexported(fq)
                        if fq is not None:
                            candidates.append(fq)
            elif len(parts) == 1 and parts[0] in self.func_aliases:
                if not self._is_shadowed(parts[0], at):
                    candidates.extend(sorted(self.func_aliases[parts[0]]))
            elif len(parts) == 1 and self.star_modules:
                starred = self._star_imported_fq(parts[0], at)
                if starred:
                    candidates.append(starred)
            held: "list[tuple[set[str], list[str]]]" = []
            if not isinstance(cur, ast.Name):
                if parts:
                    classes = self._instances_named_by(cur, at)
                    held.append((classes, parts))
                    for family in (c for c in classes if c.startswith("<")):
                        for k in range(len(parts)):
                            path = ".".join([family] + parts[:k])
                            if path in self.instance_aliases:
                                held.append((self.instance_aliases[path], parts[k:]))
            elif self.instance_aliases:
                for root in self._roots(parts[0]):
                    for k in range(1, len(parts)):
                        path = ".".join([root] + parts[1:k])
                        if path in self.instance_aliases and not (
                            k == 1 and root == parts[0] and self._is_shadowed(parts[0], at)
                        ):
                            held.append((self.instance_aliases[path], parts[k:]))
            for classes, rest in held:
                for cls in sorted(classes):
                    fq = ".".join([cls] + rest)
                    # Only a method the table places counts: `s.mount(...)` sends nothing.
                    if (
                        fq in _NETWORK_DESTINATION_ARG or fq in _UPLOAD_HTTP_METHODS
                    ) and fq not in candidates:
                        candidates.append(fq)
            for fq in list(candidates):
                top = _reexported(fq)
                if top is not None and top not in candidates:
                    candidates.append(top)
            return candidates

        def _unwrapped_url_arg(self, node: ast.AST) -> ast.AST:
            """`urlopen(Request(url))` carries the destination one call further in, so read it
            there, but only once the callee is PROVEN to be `urllib.request.Request`.

            Trusting any callee spelled `Request` reads the wrong value: a local
            `def Request(_): return "https://evil.example/x"` made the screen check the
            allowlisted argument while the runtime call sent the request somewhere else. An
            unproven callee is left wrapped, so it reads as a call rather than a literal and the
            fail-closed rule refuses it.
            """
            if isinstance(node, ast.Call) and node.args:
                if "urllib.request.Request" not in self._fq_candidates(node.func):
                    return node
                cur = node.func
                while isinstance(cur, ast.Attribute):
                    cur = cur.value
                root = cur.id if isinstance(cur, ast.Name) else None
                if root is None or root in self.rebound_anywhere or "*" in self.rebound_anywhere:
                    return node
                return node.args[0]
            return node

        def visit_Call(self, node):
            if self.collecting:
                self.pending_calls.append((node, tuple(self.scope_stack), tuple(self.self_names)))
                func = node.func
                if isinstance(func, ast.Attribute) and (
                    self._is_environ(func.value)
                    or (
                        func.attr == "putenv"
                        and isinstance(func.value, ast.Name)
                        and func.value.id in self.os_names
                    )
                ):
                    if func.attr in ("setdefault", "putenv", "__setitem__") and len(node.args) >= 2:
                        self._record_env_proxy(node.args[0], node.args[1])
                    elif func.attr in ("update", "__ior__"):
                        for arg in node.args:
                            self._record_env_mapping(arg)
                        for kw in node.keywords:
                            if kw.arg is None:
                                self._record_env_mapping(kw.value)
                            else:
                                self._record_env_proxy(ast.Constant(value = kw.arg), kw.value)
                elif isinstance(func, ast.Attribute) and func.attr in (
                    "update",
                    "setdefault",
                    "__setitem__",
                    "__ior__",
                ):
                    if isinstance(func.value, ast.Name):
                        self._record_dict_mutation(
                            func.value.id, func.attr, node.args, node.keywords
                        )
                    args = node.args
                    if func.attr in ("setdefault", "__setitem__"):
                        args = [] if args and _is_no_proxy(args[0]) else args[1:2]
                    for value in [
                        *args,
                        *(kw.value for kw in node.keywords if kw.arg != "no_proxy"),
                    ]:
                        self._record_proxy(func.value, value, mutated = True)
                self.generic_visit(node)
                return
            # Resolution may only add candidates: the alias map is not scope aware, so check both
            # spellings and let the recognised one decide.
            fq_candidates = self._fq_candidates(
                node.func, (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
            )
            # Root-name set lookup first rejects ordinary calls cheaply.
            recognised = (
                [
                    c
                    for c in fq_candidates
                    if c.partition(".")[0] in _NETWORK_ROOTS
                    and any(c.startswith(p) for p in _NETWORK_FQ_PREFIXES)
                ]
                if network_possible
                else []
            )
            fq = recognised[0] if recognised else (fq_candidates[0] if fq_candidates else "")

            chooser = node.func
            if (
                network_possible
                and isinstance(chooser, ast.Call)
                and isinstance(chooser.func, ast.Name)
                and chooser.func.id == "getattr"
                and len(chooser.args) >= 2
                and _constant_getattr(chooser) is None
            ):
                at = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
                owner = chooser.args[0]
                if self._modules_named_by(owner, at) or self._instances_named_by(owner, at):
                    network_calls.append(
                        {
                            "type": "unreadable_host_blocked",
                            "line": getattr(node, "lineno", -1),
                            "description": (
                                "Blocked: network call is chosen at runtime; "
                                "call the function by name"
                            ),
                        }
                    )

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

            if (
                not recognised
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "connect"
                and node.args
            ):
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

            if recognised:
                # 1) Upload-shape check against every recognised candidate.
                if any(_call_is_upload_shape(node, c) for c in recognised):
                    network_calls.append(
                        {
                            "type": "upload_blocked",
                            "line": getattr(node, "lineno", -1),
                            "description": ("Blocked: file upload disallowed in sandbox"),
                        }
                    )

                # 2) Extract hosts once per value the argument can hold.
                hosts: list[str] = []
                unreadable = False
                # Read each candidate with its own signature: one name can carry candidates with the URL in
                # different positions.
                specs = [
                    _NETWORK_DESTINATION_ARG[c] for c in recognised if c in _NETWORK_DESTINATION_ARG
                ]
                destinations: "list[tuple[ast.AST, bool, str]]" = []
                for index, keywords, kind in specs:
                    if index is not None and len(node.args) > index:
                        found = node.args[index]
                    else:
                        found = next(
                            (kw.value for kw in node.keywords or [] if kw.arg in keywords), None
                        )
                    if found is not None and not (
                        isinstance(found, ast.Constant) and found.value is None
                    ):
                        destinations.append((found, True, kind))
                routed = list(node.keywords or [])
                for fq, positions in _ROUTE_POSITIONS.items():
                    if fq in recognised:
                        for index, arg in enumerate(node.args):
                            if isinstance(arg, ast.Starred):
                                routed.append(ast.keyword(arg = "sock", value = _UNREADABLE))
                                break
                            if index in positions:
                                routed.append(ast.keyword(arg = positions[index], value = arg))
                for kw in routed:
                    if isinstance(kw.value, ast.Constant) and kw.value.value is None:
                        continue
                    for prefix, routes in _ROUTE_KEYWORDS.items():
                        if kw.arg in routes and any(c.startswith(prefix) for c in recognised):
                            if kw.arg == "tunnel":
                                destinations.append((kw.value, True, "host"))
                            else:
                                destinations.append((_UNREADABLE, True, "host"))
                    if kw.arg == "connect_kwargs" and any(
                        c.startswith("fabric.") for c in recognised
                    ):
                        keys = list(kw.value.keys) if isinstance(kw.value, ast.Dict) else []
                        known = isinstance(kw.value, ast.Dict)
                        if isinstance(kw.value, ast.Name):
                            name = kw.value.id
                            known = name in self.dict_literals and name not in self.dict_mutated
                            keys = [k for d in self.dict_literals.get(name, ()) for k in d.keys]
                            keys += [k for k, _v in self.dict_entries.get(name, ())]
                        if not known or any(
                            not isinstance(k, ast.Constant) or k.value == "sock" for k in keys
                        ):
                            destinations.append((_UNREADABLE, True, "host"))
                proxies = [kw.value for kw in node.keywords or [] if kw.arg in _PROXY_KEYWORDS]
                if _reads_proxy_environment(node, recognised):
                    proxies += self.env_proxies
                if "urllib.request.ProxyHandler" in recognised and node.args:
                    proxies.append(node.args[0])
                if isinstance(node.func, ast.Attribute):
                    owners = {c.rpartition(".")[0] for c in recognised}
                    linked = set()
                    for receiver in self._path_variants(node.func.value):
                        linked |= self._linked_paths(receiver)
                    for path in linked:
                        proxies.extend(
                            value
                            for attr, value in self.receiver_destinations.get(path, ())
                            if owners & _DESTINATION_ATTRS[attr]
                        )
                for proxy in proxies:
                    if isinstance(proxy, ast.Name) and proxy.id in self.dict_literals:
                        if proxy.id in self.dict_mutated:
                            unreadable = True
                        entries = [
                            (k, v)
                            for literal in self.dict_literals[proxy.id]
                            for k, v in zip(literal.keys, literal.values)
                        ] + list(self.dict_entries.get(proxy.id, ()))
                        if any(k is None for k, _v in entries):
                            unreadable = True
                        values = [v for k, v in entries if k is not None and not _is_no_proxy(k)]
                    elif isinstance(proxy, ast.Dict):
                        if None in proxy.keys:
                            unreadable = True
                        values = [
                            v for k, v in zip(proxy.keys, proxy.values) if not _is_no_proxy(k)
                        ]
                    else:
                        values = [proxy]
                    for value in values:
                        if not (isinstance(value, ast.Constant) and value.value is None):
                            destinations.append((value, True, "proxy"))
                if specs:
                    # A splat can carry the destination unseen.
                    if any(isinstance(a, ast.Starred) for a in node.args or []) or any(
                        kw.arg is None for kw in node.keywords or []
                    ):
                        unreadable = True
                elif node.args:
                    destinations.append((node.args[0], False, "url"))
                for destination, fails_closed, kind in destinations:
                    a0 = self._unwrapped_url_arg(destination)
                    is_tuple = isinstance(a0, ast.Tuple)
                    read = None if is_tuple and not a0.elts else (a0.elts[0] if is_tuple else a0)
                    candidates = (
                        [("", False)]
                        if read is None
                        else _literal_str_prefixes(read, self.literal_names)
                    )
                    for head, whole in candidates:
                        host = None
                        if is_tuple or kind == "host":
                            if whole and head.strip():
                                host = head.strip()
                        else:
                            # Clients strip leading whitespace and drop tab/CR/LF before parsing (requests 2.34.2).
                            reading = re.sub(r"[\t\r\n]", "", head).lstrip()
                            # aiohttp 3.14.3 treats `//host/x` as absolute and connects to `host`.
                            m = re.match(r"^(?:\w+:)?//([^/?#]+)", reading)
                            # A literal truncated after the first `/?#` still names the host in full.
                            if m and (whole or reading[m.end(1) :]):
                                host = m.group(1)
                            elif not m and kind == "proxy" and whole and reading.strip():
                                host = reading.strip()
                        relative = (
                            kind == "url"
                            and not is_tuple
                            and re.match(r"^\s*/[^/]", head) is not None
                        )
                        if host is None:
                            unreadable = unreadable or (fails_closed and not whole and not relative)
                        else:
                            hosts.append(host)

                # 3) A recognised call with an unreadable host is untrusted, not absent.
                if unreadable and specs and (node.args or node.keywords):
                    network_calls.append(
                        {
                            "type": "unreadable_host_blocked",
                            "line": getattr(node, "lineno", -1),
                            "description": (
                                "Blocked: network destination is not a literal the sandbox "
                                "can check; write the allowed host out in full"
                            ),
                        }
                    )
                if any(_is_metadata_host(h) for h in hosts):
                    network_calls.append(
                        {
                            "type": "metadata_host_blocked",
                            "line": getattr(node, "lineno", -1),
                            "description": "Blocked: cloud-metadata host",
                        }
                    )
                elif any(not _is_trusted_host(h) for h in hosts):
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

    _network_visitor = NetworkAndIoVisitor()
    if network_possible:
        _network_visitor.visit(tree)
        _network_visitor.resolve_flows()
    _network_visitor.collecting = False
    _network_visitor.visit(tree)

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
    """Validate code safety via static analysis. Returns an error message string if the code is
    unsafe, or None if OK."""
    safe, info = _check_signal_escape_patterns(code)
    if not safe:
        # Let SyntaxError through so the child reports a normal traceback.
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
    """Record a tool subprocess for the startup sweep. macOS has no parent-death signal, so a force
    quit mid-call would otherwise leave a session-leading tool (and whatever it spawned) with
    nothing able to find it. Best-effort: a failure here must never break a tool call."""
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

    Captured right after ``Popen`` so a later ``poll()`` / ``wait()`` that reaps the leader cannot
    make ``os.getpgid(proc.pid)`` fail first.

    Windows has no process groups, so capture the wrapper pid instead, tagged for
    ``_killpg_captured`` to reach with ``taskkill /T``; returning ``None`` there left a payload that
    outlived its wrapper unsignalled.
    """
    if os.name == "nt":
        job = _windows_job_capture(proc)
        if job is not None:
            return ("windows-job", job)
        # No job: fall back to pid plus creation-time identity, since a bare pid can be recycled.
        return ("windows-tree", proc.pid, _windows_pid_identity(proc.pid))
    if os.name != "posix" or not hasattr(os, "getpgid"):
        return None
    try:
        return os.getpgid(proc.pid)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        return None


class _WindowsToolJob:
    """A job object holding one tool call's process tree. Windows has no process groups, and
    ``taskkill`` cannot reach a tree whose root has already exited, which is exactly the case
    this capture exists for. The job stays a valid handle on every descendant either way. Created
    without kill-on-close, so releasing it never kills a process that outlived the call."""

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
    """Put ``proc`` in its own job. ``None`` when that is not possible, leaving the pid-based
    fallback."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        H, BOOL, UINT = wintypes.HANDLE, wintypes.BOOL, wintypes.UINT
        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        # Explicit widths, or ctypes truncates 64-bit handles to c_int.
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
        # The Popen handle avoids a pid-recycling window.
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

    Every tool call runs under a shell wrapper, and Windows has no process groups, so a bare
    ``proc.kill()`` reaps the wrapper and orphans the payload (usually the venv python), which then
    blocks `unsloth studio update`.

    ``identity`` is the creation time captured at spawn; a mismatch means the pid now belongs to
    something else, so nothing is signalled.
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
    """SIGKILL a process group captured before its leader was waited on. Once ``proc`` exits,
    ``os.getpgid(proc.pid)`` fails and ``_kill_process_tree`` short-circuits, so a stdout-holding
    grandchild that outlived the parent could not otherwise be signaled. The pre-captured setsid
    group id still targets the whole tree. On Windows the capture is a tagged pid and the
    equivalent reach is ``taskkill /T /F``. No-op when nothing was captured."""
    if pgid is None:
        return
    if isinstance(pgid, tuple):
        if pgid[0] == "windows-job":
            pgid[1].terminate()
            return
        _tag, pid, identity = pgid
        # Without a verified identity the pid may be someone else's now; the job object still cleans up.
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
    """Daemon thread that kills a process when cancel_event is set. ``pgid`` is the group id
    captured right after spawn; killing it directly reaps a stdout-holding grandchild even when
    the watcher's own ``poll()`` already reaped the leader (which makes ``_kill_process_tree``
    short-circuit)."""
    while proc.poll() is None:
        if cancel_event is not None and cancel_event.is_set():
            request_cancel = getattr(proc, "_unsloth_cancel", None)
            if request_cancel is not None and request_cancel():
                try:
                    proc.wait(timeout = 5)
                    return
                except subprocess.TimeoutExpired:
                    pass
            _killpg_captured(pgid)
            _kill_process_tree(proc)
            return
        cancel_event.wait(poll_interval) if cancel_event else None


def _appended_by_the_loop(text: str) -> float:
    """What the tool loop will add to this result after the tool has handed it back.
    `ToolCallCompletion.model_message` appends `TOOL_ERROR_NUDGE` to a result that opens with one
    of `TOOL_ERROR_PREFIXES`, after this budget has already let the body take the whole room, and
    a parallel batch of failed calls carries one nudge each. Priced in tokens like the retry
    hint, and only for the results that will really carry it."""
    try:
        from .tool_call_parser import TOOL_ERROR_NUDGE, TOOL_ERROR_PREFIXES  # noqa: PLC0415
    except Exception:  # noqa: BLE001 -- an unpriced nudge, not a failed tool call
        logger.debug("result budget: tool error nudge unavailable", exc_info = True)
        return 0.0
    # `lstrip` because `is_tool_error` does.
    if not text.lstrip().startswith(TOOL_ERROR_PREFIXES):
        return 0.0
    return _text_token_cost(TOOL_ERROR_NUDGE, _window_context_tokens())


def _truncate(
    text: str,
    limit: int | None = None,
    workdir: str | None = None,
    scope: "str | None" = "",
    hint: str = "",
    reserve_tokens: float = 0.0,
    omitted: "tuple[int, int]" = (0, 0),
) -> str:
    # Resolved per call: an import-time default freezes before any model loads.
    if limit is None:
        limit = _tool_result_char_budget()
    size = len(text) + omitted[0]
    if omitted[0]:
        limit = min(limit, _SPILL_MAX_BYTES)
    # Reserve for what the loop (error nudge) or caller appends after this result, so concatenated
    # fitted strings do not spend the room twice.
    cap, cost = limit, _appended_by_the_loop(text) + reserve_tokens
    if hint:
        # Priced in tokens and capped at half the room; dense paths cost more than their length.
        plain = _dense_char_limit(text, limit, cost)
        cost += _text_token_cost(hint, _window_context_tokens())
        with_hint = _dense_char_limit(text, limit, cost)
        if plain > 0 and (len(text) <= with_hint or with_hint * 2 >= plain):
            limit = with_hint
        else:
            limit, hint, cost = plain, "", _appended_by_the_loop(text) + reserve_tokens
    else:
        limit = _dense_char_limit(text, limit, cost)
    # Mode-neutral and byte-identical with or without an output_callback (tested invariant).
    if len(text) <= limit:
        return text + hint
    # Notice charged only now that it is certain, and only against a priced room; legacy character
    # caps always appended it past the cap.
    if _request_result_room() is not None:
        limit = _dense_char_limit(text, cap, cost + _RESULT_NOTICE_RESERVE)
    if limit <= 0 and len(_zero_room_stub(size, None, True)) >= len(text):
        # Before the spill: no file for output that is never cut.
        return text + hint
    spill, complete = _spill_full_output(text, workdir, scope)
    if limit <= 0:
        # No room: a one-line stub keeps the thread servable so the next fit can recover.
        stub = _zero_room_stub(size, spill, complete)
        return (stub if len(stub) < len(text) else text) + hint
    head, on_boundary = _head_whole_lines(text, limit)
    if spill is None:
        return (
            head
            + (
                f"\n\n... (truncated to {limit} chars for the model; {size} chars "
                "total. The full output is not retained here; any files the code wrote "
                "persist in the working directory.)"
            )
            + hint
        )
    # Name the exact paging command, or the model re-runs and truncates identically.
    if on_boundary:
        # No +1 when the head ends in a newline, or the hint skips a line.
        shown = 0 if not head else head.count("\n") + (0 if head.endswith("\n") else 1)
        total = text.count("\n") + 1 + omitted[1]
        resume = f"sed -n '{shown + 1},{shown + max(1, shown)}p' {spill}"
        where = f"showing lines 1-{shown} of {total}"
    else:
        # Cut mid-line: resume by byte offset, which is what `tail -c` counts.
        offset = len(head.encode("utf-8", "surrogatepass"))
        # Byte length of the next chunk's characters so `head -c` never splits a code point.
        chunk = text[len(head) : len(head) * 2 or None]
        span = len(chunk.encode("utf-8", "surrogatepass"))
        resume = f"tail -c +{offset + 1} {spill} | head -c {max(1, span)}"
        where = f"showing the first {len(head)} chars of {size}"
    # Always kept: the only thing telling the model its written files survive.
    common = (
        f"\n\n... (truncated to {limit} chars for the model; {where}, {size} chars "
        f"total. {_capitalise(_spill_phrase(spill, complete))}, and any files the code "
        "wrote persist in the working directory"
    )
    if not _posix_tools_available():
        # cmd-only Windows lacks sed/tail/head; do not promise paging.
        return head + common + ".)" + hint
    return head + common + f" -- continue with:\n  {resume})" + hint


def _fit_result_to_room(text, name = None):
    """Cap a tool that does not cap its own output, when this request priced its room.

    `python` and `terminal` truncate against this same budget before they return, so this is a no-op
    for them. The other tools hand their string back whole: an MCP response is unbounded, a fetched
    page or an edit receipt runs to a few thousand characters, and any of them can overflow a nearly
    full local thread and then be protected as its newest exchange, which is the failure the budget
    exists to prevent.

    No spill file: these tools have no sandbox of their own, and `edit_file` must not have a workdir
    created underneath a caller running with code execution off. So the cap comes with the plain
    notice and no paging hint.

    With no priced room (external providers, the hosted path, any loop that does not measure its own
    conversation) the text is returned untouched.
    """
    if _request_result_room() is None or not isinstance(text, str) or not text:
        return text
    # Only the model-visible part is measured; the frontend envelope must stay byte-identical
    # (a cut JSON array replays broken base64).
    body, suffix = _split_frontend_suffix(text, name)
    if not body:
        return text
    fitted = _truncate(body)
    return fitted + suffix if fitted is not body else text


def _split_frontend_suffix(text: str, name: "str | None") -> "tuple[str, str]":
    """``text`` split into what the model sees and the trailing frontend-only envelope.
    `strip_result_for_model` is the same function the replay path uses, so the split follows its
    validation exactly: a result that merely mentions a sentinel keeps it in the body and is
    capped with it, which is the conservative half."""
    from .tool_loop_controller import strip_result_for_model

    try:
        # Unredacted so the strip removes only the suffix; the model copy is masked in model_message.
        body = strip_result_for_model(text, name, redact = False)
    except Exception:
        logger.debug("frontend suffix split failed", exc_info = True)
        return text, ""
    if not isinstance(body, str) or not text.startswith(body):
        return text, ""
    return body, text[len(body) :]


MAX_TOOL_TEXT_CHARS = _env_int("UNSLOTH_TOOL_RESULT_HARD_CAP_CHARS", 256_000)
_TOOL_TEXT_READERS = frozenset({"terminal", "python"})


def _hard_cap_chars() -> int:
    """Never below the window-aware cap plus its notice, so output `_truncate` already cut (and
    spilled) passes through with its own spill reference intact."""
    return max(MAX_TOOL_TEXT_CHARS, _MAX_OUTPUT_CHARS + 4_000)


def _tool_text_notice_head() -> str:
    return f"\n\n... (tool result truncated to {_hard_cap_chars():,} chars for the model;"


def _tool_text_search_hint(path: str, readers: "frozenset[str]") -> str:
    ways = []
    if "terminal" in readers and _posix_tools_available():
        ways += [f"grep -n 'pattern' {path}", f"sed -n '1,200p' {path}"]
    elif "terminal" in readers:
        ways.append(f'findstr /n "pattern" {path.replace("/", chr(92))}')
    if "python" in readers:
        ways.append(f"open({path!r}) in python")
    return "Search it instead of re-running the call, e.g. " + ", or ".join(ways)


def cap_tool_text(
    text: str,
    *,
    session_id: "str | None" = None,
    thread_id: "str | None" = None,
    readers: "frozenset[str]" = frozenset(),
) -> str:
    """Unconditional floor (``UNSLOTH_TOOL_RESULT_HARD_CAP_CHARS``); spills when a reader tool exists."""
    limit = _hard_cap_chars()
    if len(text) <= limit:
        return text
    head = _head_whole_lines(text, limit)[0]
    readers = readers & _TOOL_TEXT_READERS
    if readers and session_id and _spill_scope(session_id, thread_id) is not None:
        try:
            workdir = _get_workdir(session_id)
        except Exception:  # noqa: BLE001 -- no sandbox means the plain notice
            logger.debug("tool text spill: no workdir", exc_info = True)
            workdir = None
        from .tool_loop_controller import redact_studio_credentials  # noqa: PLC0415

        spill, complete = _spill_full_output(
            redact_studio_credentials(text), workdir, _spill_scope(session_id, thread_id)
        )
        if spill is not None:
            return (
                head
                + f"{_tool_text_notice_head()} {_spill_phrase(spill, complete)} in the working "
                f"directory. {_tool_text_search_hint(spill, readers)}.)"
            )
    return head + f"{_tool_text_notice_head()} the full output is not retained in model context.)"


def _head_whole_lines(text: str, limit: int) -> "tuple[str, bool]":
    """``text`` cut to at most ``limit`` characters, and whether it ended on a line break.

    Whole lines where possible, so the hint can name a line that resumes exactly where this stopped;
    a mid-line cut would repeat or lose one, and on a file printed verbatim that is a line of the
    user's own code.

    The flag is not decoration. One enormous line (minified JS, base64) has no boundary to cut on,
    and a line number would then name the line AFTER the one the reader is halfway through, skipping
    the rest of it. On single-line output that returns nothing at all.
    """
    head = text[:limit]
    cut = head.rfind("\n")
    if cut > 0 and cut >= limit // 2:
        return head[:cut], True
    return head, head.endswith("\n")


def _posix_tools_available() -> bool:
    """Whether the shell these tools run in has sed/tail/head. `_get_shell_cmd` falls back to `cmd
    /c` on a Windows host with no trusted bash, and none of those exist there, so a hint naming
    them is a command the model cannot run. The spill is still worth naming; the command is not."""
    if sys.platform != "win32":
        return True
    return _windows_bash() is not None


# Dot-directory so `_snapshot_workdir_files` skips it and no phantom download card appears.
_SPILL_DIR = ".unsloth_tool_output"
_SPILL_KEEP = 20
# Bounds disk use; the subprocess file-size limit does not apply to piped output.
_SPILL_MAX_BYTES = 8 * 1024 * 1024

# Chunked encoding yields the same byte stream as encoding the whole string.
_SPILL_HASH_CHUNK_CHARS = 1 << 20


def _digest_and_head(text: str, max_bytes: int) -> "tuple[str, int, bytes]":
    """``(digest, encoded length, the first max_bytes of it)``, in one bounded pass. Encoding the
    whole result to hash it and again to cut it puts two more copies of it through memory, and at
    most `max_bytes` of the second is ever written. The output this runs on is by definition the
    output that did not fit: `cat` of a file the model just wrote can be hundreds of megabytes,
    and spending it twice more inside the backend risks the stall or the OOM instead of the
    bounded answer this path exists to return."""
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
# Exactly the generated names: the prune deletes matches in what may be the user's directory.
_SPILL_NAME_RE = re.compile(r"[0-9a-f]{12}\.txt")
# Ownership is recorded, not inferred from names: the directory may be the user's.
_SPILL_RECORD_HEADER = "unsloth-studio tool output "
# Per-root lock: chats in a project share a sandbox and the manifest is read-modify-write.
_SPILL_LOCKS: "dict[str, threading.Lock]" = {}
_SPILL_LOCKS_GUARD = threading.Lock()


def _spill_lock(root: str) -> "threading.Lock":
    key = os.path.realpath(root)
    with _SPILL_LOCKS_GUARD:
        return _SPILL_LOCKS.setdefault(key, threading.Lock())


def _spill_records_dir() -> str:
    """Where the spill manifests live: Unsloth's own storage, NOT the sandbox. The sandbox is a
    directory tool code writes to, so nothing kept inside it can be evidence about the sandbox. A
    marker file there was replaceable by a link, and once it is a plain file the model can
    rewrite its contents and name the user's own files as Unsloth's, which turns the cleanup into
    a delete and the prune into an unlink. Held beside the other records this file already keeps
    outside the sandboxes."""
    try:
        from utils.paths.storage_roots import account_path  # noqa: PLC0415
        return str(account_path("tool-output-records"))
    except Exception:
        return os.path.join(
            os.path.dirname(os.path.realpath(sandbox_root())), "tool-output-records"
        )


def _spill_record_path(root: str) -> str:
    """The record for one spill root, named by a digest of its real path."""
    digest = hashlib.sha256(os.path.realpath(root).encode("utf-8", "surrogatepass")).hexdigest()
    return os.path.join(_spill_records_dir(), f"{digest[:24]}.txt")


def _spill_identity(root: str) -> "str | None":
    """The directory's device and inode, which is what the record claims ownership OF. Tool code can
    delete `.unsloth_tool_output` and make its own in the same place. That is a different
    directory with the same path, and a record that only knew the path would hand the new one's
    contents to the prune."""
    try:
        stat = os.lstat(root)
    except OSError:
        return None
    if not os.path.isdir(root) or os.path.islink(root):
        return None
    return f"{stat.st_dev}:{stat.st_ino}"


def _own_spill_root(root: str) -> bool:
    """Whether the spill directory is one this process made, creating it if it is absent. Ownership
    is recorded outside the sandbox (`_spill_records_dir`) and is of a specific directory, not of
    a path: a `.unsloth_tool_output` that came with the sandbox, or one tool code deleted and
    recreated, has no matching record and is left alone entirely."""
    if os.path.islink(root):
        return False
    try:
        # Check, create and first record in one locked step, or the loser overwrites the winner's record.
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
                    return False
                os.makedirs(_spill_records_dir(), exist_ok = True)
                _write_spill_manifest(root, {}, identity = identity)
                recorded = identity
            return recorded == identity
    except OSError:
        logger.debug("tool result spill ownership check failed", exc_info = True)
        return False


# Windows lacks O_DIRECTORY and dir_fd; the path-based write stands there.
_DIR_FD_WRITES = (
    hasattr(os, "O_DIRECTORY")
    and os.open in getattr(os, "supports_dir_fd", set())
    and os.link in getattr(os, "supports_dir_fd", set())
    and os.unlink in getattr(os, "supports_dir_fd", set())
)


def _write_spill_file(target_dir: str, name: str, body: str) -> "str | None":
    """Write one spill into ``target_dir``, without following a link at any point.

    The directory is opened ONCE, O_NOFOLLOW, and every step after that is relative to that
    descriptor. Checking the path and then writing to it by name is a race a shared project sandbox
    can lose: another call can replace the directory with a symlink in between, and both the create
    and the rename would follow it, putting this output outside the sandbox with the backend's own
    permissions.

    Still written to a fresh O_EXCL file and installed under the real name rather than opened over
    whatever is there: the spill name comes from content the model produced, so it can predict it
    and pre-create it, as a symlink or as a hard link sharing an inode with some file elsewhere.

    Installed with `os.link`, which fails when the name is taken, rather than a rename, which on
    POSIX replaces silently. The caller checks the destination first, but between that check and
    this write another call sharing the workspace can put a file there, and a rename would then
    destroy it.

    Returns the stamp of what was installed, or None if nothing was. The stamp is taken here rather
    than re-read from the path afterwards, because by then another call can have replaced the file
    and the record would name its content as Unsloth's.
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
        # link fails with EEXIST; rename would silently replace a name taken meanwhile.
        os.link(tmp_name, name, src_dir_fd = dir_fd, dst_dir_fd = dir_fd)
        _quiet_unlink(tmp_name, dir_fd = dir_fd)
        # Through the same descriptor, so it is the file just linked.
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


_ATTACHMENTS_DIR = ".unsloth_attachments"
_ATTACHMENT_PREFIX_LEN = 12
_ATTACHMENT_NAME_BYTES = 80
_UNSAFE_NAME_CHARS = re.compile(r'[\x00-\x1f\x7f/\\:*?"<>|]')
# Windows device names stay reserved with any extension (NUL.tar.gz is NUL).
_RESERVED_NAME = re.compile(r"(?:CON|PRN|AUX|NUL|COM\d|LPT\d)", re.IGNORECASE)


def sandbox_attachment_path(sha256: str, name: str) -> str:
    """Mirrored by sandboxAttachmentPath in the frontend's sandbox-attachments.ts."""
    base = _UNSAFE_NAME_CHARS.sub("_", name or "").strip(" .") or "attachment"
    # In bytes: filesystems cap a name at 255, and macOS stores decomposed text that can triple it.
    if len(base.encode()) > _ATTACHMENT_NAME_BYTES:
        stem, ext = os.path.splitext(base)
        ext = ext if len(ext.encode()) <= 16 else ""
        room = _ATTACHMENT_NAME_BYTES - len(ext.encode())
        # strip again so the basename the frontend sends back derives the same path.
        base = (stem.encode()[:room].decode("utf-8", "ignore").rstrip(" .") or "attachment") + ext
    if _RESERVED_NAME.fullmatch(base.split(".", 1)[0].rstrip(" ")):
        base = "_" + base
    return f"{_ATTACHMENTS_DIR}/{sha256[:_ATTACHMENT_PREFIX_LEN]}/{base}"


def materialize_sandbox_attachments(
    session_id: "str | None", attachments: "list[tuple[str, str]]"
) -> None:
    """copy chat attachment originals into the sandbox, preserving existing copies so edits survive."""
    from core import chat_originals
    with _session_in_flight(session_id):
        workdir = _get_workdir(session_id)
        missing = [
            (sha256, name, chat_originals.originals_dir() / sha256)
            for sha256, name in attachments
            if not os.path.lexists(os.path.join(workdir, sandbox_attachment_path(sha256, name)))
        ]
        missing = [entry for entry in missing if entry[2].is_file()]
        if not missing:
            return
        # register the copy as a call so concurrent chats sharing the workdir cannot claim it.
        token = _call_started(workdir)
        try:
            for sha256, name, source in missing:
                try:
                    _install_attachment_copy(workdir, sandbox_attachment_path(sha256, name), source)
                except (OSError, ValueError):
                    logger.warning(
                        "could not copy attachment %s into the sandbox", sha256, exc_info = True
                    )
        finally:
            _call_finished(token)


def _install_attachment_copy(workdir: str, relative: str, source: Path) -> None:
    """match the spill writer: follow no links; `os.link` never replaces existing names."""
    *dirs, name = relative.split("/")
    tmp = f".tmp-{uuid.uuid4().hex[:12]}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if not _DIR_FD_WRITES:
        target = workdir
        for part in dirs:
            target = os.path.join(target, part)
            if os.path.islink(target):
                return
            os.makedirs(target, mode = 0o700, exist_ok = True)
        if os.path.realpath(target) != os.path.join(os.path.realpath(workdir), *dirs):
            return
        if os.path.lexists(os.path.join(target, name)):
            return
        tmp = os.path.join(target, tmp)
        try:
            with (
                open(source, "rb") as src,
                os.fdopen(os.open(tmp, flags | getattr(os, "O_NOFOLLOW", 0), 0o600), "wb") as out,
            ):
                shutil.copyfileobj(src, out, 1 << 20)
            with contextlib.suppress(FileExistsError):
                os.link(tmp, os.path.join(target, name))
        finally:
            _quiet_unlink(tmp)
        return
    fds = [os.open(workdir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)]
    try:
        for part in dirs:
            with contextlib.suppress(FileExistsError):
                os.mkdir(part, 0o700, dir_fd = fds[-1])
            fds.append(os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd = fds[-1]))
        with contextlib.suppress(FileNotFoundError):
            os.stat(name, dir_fd = fds[-1], follow_symlinks = False)
            return
        try:
            with (
                open(source, "rb") as src,
                os.fdopen(os.open(tmp, flags, 0o600, dir_fd = fds[-1]), "wb") as out,
            ):
                shutil.copyfileobj(src, out, 1 << 20)
            with contextlib.suppress(FileExistsError):
                os.link(tmp, name, src_dir_fd = fds[-1], dst_dir_fd = fds[-1])
        finally:
            _quiet_unlink(tmp, dir_fd = fds[-1])
    finally:
        for fd in fds:
            os.close(fd)


def _is_attachment_copy(sandbox: str, parent: str, name: str) -> bool:
    prefix = os.path.basename(parent)
    path = os.path.join(parent, name)
    if (
        os.path.dirname(parent) != os.path.join(sandbox, _ATTACHMENTS_DIR)
        or not re.fullmatch(r"[0-9a-f]{12}", prefix)
        or os.path.islink(path)
    ):
        return False
    digest = _file_digest(path)
    return digest is not None and digest.startswith(prefix)


def _forget_spill_record(path: str) -> None:
    """Drop the record for a sandbox that has been removed. See `_spill_records_dir`."""
    try:
        os.remove(path)
    except OSError:
        pass


def _spill_stamp(path: str) -> "str | None":
    """What a spill looked like when it was written: device, inode, size, mtime, ctime. A recorded
    PATH is not the file: tool code can write its own content over one, in place, keeping the
    inode. mtime alone is not enough either, since `os.utime` can put it back and a
    coarse-grained filesystem can leave it unchanged on its own; ctime moves on any write to the
    file OR its metadata and cannot be set back from userspace, so restoring the mtime is itself
    a change this sees."""
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

    Opened O_NOFOLLOW and O_NONBLOCK and checked through the DESCRIPTOR, not the path. The stamp was
    taken a moment ago and this runs in the sandbox's own directory: between the two, tool code can
    put a symlink at the name, or a FIFO, or a device. A plain open would follow the first and block
    forever on the others, and this is called synchronously by the prune and by chat deletion,
    neither of which has a timeout.

    ``expected_size`` refuses anything that is not the size the record remembers, so a file swapped
    for an enormous one is not hashed before it is rejected.
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
    """Whether ``path`` is still the file this wrote, rather than one written over it. Both the
    stamp and the CONTENT, because the stamp is only evidence about metadata: the content is what
    says this file is still the output this process put there, and it is the last word before
    anything is deleted. Read only after the cheap check passes, so the cost falls on the handful
    of files that already look like ours."""
    relative = os.path.relpath(path, root).replace(os.sep, "/")
    recorded = owned.get(relative)
    if recorded is None or recorded[0] != _spill_stamp(path):
        return False
    return recorded[1] == _file_digest(path, _stamp_size(recorded[0]))


def _spill_record(root: str) -> "tuple[str | None, dict[str, tuple[str, str]]]":
    """The recorded identity of ``root`` and the spills written into it. ``(None, {})`` when there
    is no record, which is the answer for any directory this process did not create. An
    unreadable or half-written record reads the same way, which retains too much rather than
    deleting something that was never ours."""
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
    """Append one written spill, with the stamp and digest of what was INSTALLED. Not re-read from
    the path: between the install and this, another call sharing the sandbox can replace the
    file, and stating the path then records that call's content as Unsloth's, which a later prune
    or cleanup would delete. The writer knows what it put there, so it says so."""
    try:
        with _spill_lock(root):
            with open(_spill_record_path(root), "a", encoding = "utf-8") as handle:
                handle.write(f"{relative}\t{stamp}\t{digest}\n")
    except OSError:
        logger.debug("tool result spill record append failed", exc_info = True)


def _spill_scope(session_id: "str | None", thread_id: "str | None") -> "str | None":
    """Where this call's spills belong, or None for retain nothing.

    Nothing is retained in a sandbox that more than one chat runs in. A project's chats share one
    session by design (`project_session_id`) and a call with no session lands in the shared
    `_default` one, and in both the sandbox is a directory every sibling chat's model has a terminal
    in: a sub-directory is not access control, it is a name. Writing a result there puts output that
    existed only in one chat's response on disk where the next chat can read it, and lets that chat
    prune the files this one was told to page through.

    So the trade is stated rather than hidden: in a project, a large result is truncated with a
    notice and no continuation. Only a chat with a sandbox of its own gets paging. ``thread_id`` is
    taken and unused for that reason: it identifies the chat, which is not the thing that has to be
    separate.
    """
    if not session_id or session_id.startswith(_PROJECT_SESSION_PREFIX):
        return None
    return ""


def _spill_phrase(spill: str, complete: bool) -> str:
    """How the notice names the spill, which depends on whether all of it got there. Lower case, and
    capitalised by the caller that needs it: the phrase ends in a path, and case-folding a whole
    sentence to fit it into another would fold that too."""
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
    """Write the result into the sandbox; return its relative path and whether it is whole. ``(None,
    True)`` whenever it cannot be done: no workdir, a read-only mount, a full disk, a path that
    is not what it claims to be. The caller then falls back to the plain notice, because a hint
    naming a file that is not there is worse than admitting the output is gone."""
    if not workdir or not os.path.isdir(workdir) or scope is None:
        return None, True
    try:
        if not _own_spill_root(os.path.join(workdir, _SPILL_DIR)):
            return None, True
        relative = f"{_SPILL_DIR}/{scope}" if scope else _SPILL_DIR
        target_dir = os.path.join(workdir, *relative.split("/"))
        # The spill dir may be a symlink a tool made; refuse rather than write outside the sandbox.
        if any(
            os.path.islink(os.path.join(workdir, *relative.split("/")[: n + 1]))
            for n in range(len(relative.split("/")))
        ):
            return None, True
        os.makedirs(target_dir, exist_ok = True)
        expected = os.path.join(os.path.realpath(workdir), *relative.split("/"))
        if os.path.realpath(target_dir) != expected:
            return None, True
        # Named from content so the notice is byte-identical with and without streaming (tested), and
        # repeats reuse one spill. Bounded before writing.
        digest, spilled_bytes, head = _digest_and_head(text, _SPILL_MAX_BYTES)
        name = f"{digest}.txt"
        complete = spilled_bytes <= _SPILL_MAX_BYTES
        body = text if complete else head.decode("utf-8", "ignore")
        # newline="" so bytes on disk match the measured byte offsets.
        path = os.path.join(target_dir, name)
        # Fresh O_EXCL file moved into place, never O_TRUNC: a predicted name could be a hard link to
        # a file outside the sandbox. `_is_recorded_spill` says whether the path is still ours.
        path = os.path.join(target_dir, name)
        if os.path.exists(path):
            # A recorded spill at this digest name already holds this content: reuse it.
            if _is_recorded_spill(
                os.path.join(workdir, _SPILL_DIR),
                path,
                _spill_manifest(os.path.join(workdir, _SPILL_DIR)),
            ):
                return f"{relative}/{name}", complete
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
        # Relative so the absolute sandbox path never reaches the model.
        return f"{relative}/{name}", complete
    except Exception:
        logger.debug("tool result spill failed", exc_info = True)
        return None, True


def _spill_files(root: str, directory: str, owned: "set[str]") -> "list[str]":
    """The spills in one directory: the ones ``owned`` records this process having written.
    Everything else there is the user's, whatever it is called. Links are never spills."""
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

    Moved to a private name first. A rename is atomic, so from that point the inode this verifies is
    the inode this deletes: a sandbox writer that replaces the original name afterwards replaces
    nothing that is on its way out. Verifying and then unlinking by name cannot promise that,
    because the manifest lock orders Unsloth's own threads and the thing racing here is the sandbox.

    Checked again under the private name, and put back if it no longer matches, since at that point
    it is not the file this recorded and is not this to delete. The rename itself moves ctime, which
    is ours to move, so the second check compares the device, inode, size and mtime, and the
    content.

    Nothing but a regular file with the recorded identity is moved at all: a directory or a FIFO
    left at the name by the sandbox is rejected through its own descriptor before the rename, so the
    restore below is never asked to put back a kind of thing `os.link` cannot. That leaves only the
    vanishing window between the check and the rename, and the restore handles it by falling back to
    a rename when the name is free again.
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
    """Whether ``path`` is right now the regular file ``stamp`` recorded. Through a descriptor
    rather than the path, and for the same reasons `_file_digest` does it that way: O_NOFOLLOW
    refuses a symlink dropped at the name and O_NONBLOCK refuses to hang on a FIFO or a device. A
    directory opens, and is rejected by the mode check. ctime is deliberately not compared: the
    caller compares it under the private name, where the rename it performs has already moved it."""
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
    """Put back something the prune moved but must not delete. `os.link` first, because it refuses
    to overwrite: if the sandbox has taken the name again in the meantime, the file it put there
    is not this to replace. A hard link cannot be made to a directory, so when it fails and the
    name is in fact free, the rename that moved this here is reversed instead, which works
    whatever was moved. Neither can promise the name is still free at the instant it runs, so the
    last resort is to leave the data under the private name and say where it went, rather than
    unlink it."""
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
    logger.warning("tool result spill prune: kept a swapped file as %s", private)


def _prune_spills(target_dir: str, root: "str | None" = None) -> None:
    """Bound this scope by count and the whole spill tree by bytes.

    A long session prints many large results and each one is retained; without this the sandbox
    grows without bound for output the model has almost certainly finished paging through.

    The byte budget is enforced across ``root`` rather than per directory. A project's chats each
    get their own scope under one shared sandbox, so a per-directory limit is really a per-chat
    limit and a project with many tool-using chats multiplies it by however many there are.
    """
    root = root or target_dir
    try:
        # Held across read and rewrite so a concurrent append is not dropped.
        with _spill_lock(root):
            _prune_spills_locked(target_dir, root)
    except Exception:
        logger.debug("tool result spill prune failed", exc_info = True)


def _prune_spills_locked(target_dir: str, root: str) -> None:
    try:
        identity, owned = _spill_record(root)
        if identity is None or identity != _spill_identity(root):
            return
        removed = set()
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
            # The newest is always kept: the returned notice names it.
            if kept == 0 or total + size <= _SPILL_MAX_TOTAL_BYTES:
                kept, total = kept + 1, total + size
                continue
            if _unlink_verified_spill(root, path, owned):
                removed.add(path)
        _write_spill_manifest(
            root,
            {
                name: stamp
                for name, stamp in owned.items()
                if os.path.join(root, *name.split("/")) not in removed
            },
        )
        # Remove emptied scopes so cleanup does not see the sandbox as non-empty.
        for name in sorted(os.listdir(root)):
            scope = os.path.join(root, name)
            if os.path.isdir(scope) and not os.path.islink(scope) and not os.listdir(scope):
                try:
                    os.rmdir(scope)
                except OSError:
                    pass
    except Exception:
        logger.debug("tool result spill prune failed", exc_info = True)


# ChatGPT code-interpreter path habits that do not exist here.
_MISSING_PATH_PREFIXES = (
    "/mnt/data",
    "/mnt/outputs",
    "/home/sandbox",
    "/workspace",
    "/tmp/outputs",
)

_QUOTED_ABS_PATH_RE = re.compile(r"""['"](/[^'"\n]+)['"]""")
_BASH_ABS_PATH_RE = re.compile(r"(/[^\s:'\"]+):\s*No such file or directory")

# An absolute path under the session dir is a genuine local miss; resolved like _get_workdir.


def _missing_error_lines(output: str) -> list[str]:
    """The lines that actually name a missing file (a FileNotFoundError message or a bash "No such
    file or directory"). Traceback frame lines are excluded, so an unrelated absolute path
    mentioned elsewhere in the output is never treated as the failing one."""
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
    """True when ``abs_path`` is not the working directory or under it. ``workdir`` is the
    executor's actual working directory (defaults to the sandbox root). Project-backed sessions
    run under a root OUTSIDE ``~/studio_sandbox`` (see ``_get_workdir``), so a legitimate miss
    inside a project must be judged against the real workdir, not a static sandbox root, or it is
    wrongly classed as an external habit path."""
    try:
        root = os.path.realpath(workdir or sandbox_root())
        rp = os.path.realpath(abs_path)
    except (OSError, ValueError):
        return True
    return rp != root and not rp.startswith(root + os.sep)


def _missing_path_hint(output: str, workdir: str | None = None) -> str:
    """Model-visible healing when an execution fails on an absolute path missing in the sandbox (a
    code-interpreter habit path, or one invented from the CWD). Detected on the full
    pre-truncation output; the hint echoes the failing path so the model retries with the right
    relative name."""
    error_lines = _missing_error_lines(output)
    if not error_lines:
        return ""
    abs_path = _extract_missing_abs_path(output)
    # Only on the failing-path error lines, and only when the exact path was not isolated.
    convention = any(prefix in line for line in error_lines for prefix in _MISSING_PATH_PREFIXES)
    if abs_path is not None:
        # Judge against the real workdir, so a project under such a prefix is not misdirected.
        if not _is_outside_workdir(abs_path, workdir):
            return ""
    elif not convention:
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


# Kept past the spill head so a trailing traceback reaches _missing_path_hint.
_DRAIN_TAIL_CHARS = 64 * 1024


def _drain_process_output(
    proc,
    timeout,
    output_callback,
    cancel_event = None,
    *,
    pgid = None,
) -> "tuple[str, bool, tuple[int, int]]":
    """``proc.communicate(timeout=...)`` equivalent that also streams each stdout line (in
    ``_DRAIN_TAIL_CHARS`` pieces when longer) to ``output_callback`` as it is produced.

    Returns ``(output, timed_out, omitted)``. The joined output is what ``communicate`` would
    return: the same TextIOWrapper decodes the stream, so encoding, error replacement, and newline
    translation all match. Past the spill's head only a rolling tail is kept, and ``omitted`` is the
    ``(chars, lines)`` dropped between them. On timeout the process tree is killed (mirroring the
    non-streaming path).
    With ``timeout=None`` the drain waits for EOF like ``communicate`` would, stopping early only
    when ``cancel_event`` is set.
    """
    chunks: list[str] = []
    tail: "deque[str]" = deque()
    kept = joined = tail_chars = omitted_chars = omitted_lines = 0

    # Captured before waiting so a pipe-holding grandchild can be killed after the leader is reaped.
    if pgid is None:
        pgid = _capture_process_group(proc)

    def _reader() -> None:
        nonlocal kept, joined, tail_chars, omitted_chars, omitted_lines
        try:
            # Sized reads: a newline-free stream would otherwise arrive as one unbounded "line".
            for line in iter(lambda: proc.stdout.readline(_DRAIN_TAIL_CHARS), ""):
                if kept <= _SPILL_MAX_BYTES:
                    chunks.append(line)
                    kept += len(line)
                    if len(chunks) - joined >= 1024:
                        chunks[joined:] = ["".join(chunks[joined:])]
                        joined += 1
                else:
                    tail.append(line)
                    tail_chars += len(line)
                    while tail_chars > _DRAIN_TAIL_CHARS and len(tail) > 1:
                        gone = tail.popleft()
                        tail_chars -= len(gone)
                        omitted_chars += len(gone)
                        omitted_lines += gone.count("\n")
                if output_callback is not None:
                    try:
                        output_callback(line)
                    except Exception:  # noqa: BLE001 - observer must never kill the tool
                        logger.debug("tool output_callback raised", exc_info = True)
        except (ValueError, OSError):
            pass

    reader = threading.Thread(target = _reader, daemon = True)
    reader.start()
    started_at = time.monotonic()
    timed_out = False
    try:
        proc.wait(timeout = timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(proc)
        # Also kill the pre-captured group in case the leader was reaped first.
        _killpg_captured(pgid)
        try:
            proc.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            pass
    if not timed_out:
        if timeout is not None:
            # Poll cancel_event in slices: the cancel watcher is gone once the leader exits.
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
            while reader.is_alive():
                if cancel_event is not None and cancel_event.is_set():
                    _killpg_captured(pgid)
                    break
                reader.join(timeout = 0.5)
    reader.join(timeout = 5)
    return "".join(chunks) + "".join(tail), timed_out, (omitted_chars, omitted_lines)


_MAX_REPORTED_FILES = 25
# In path segments, matching the download route, so cards never advertise refused files.
_MAX_SANDBOX_PATH_SEGMENTS = 4
_MAX_SNAPSHOT_FILES = 2000
_MAX_SNAPSHOT_DIRS = 2000


def _user_path_parts(parts: "list[str]", root: "str | None" = None) -> "list[str]":
    """The segments _MAX_SANDBOX_PATH_SEGMENTS applies to.

    The scratch container is Unsloth's, not a name the model chose, and on Windows it is what /tmp
    resolves to, so charging it a segment would drop one level of the /tmp artifacts served before
    the workdir stopped being %TEMP%.

    *root* is for callers that resolve the path afterwards. The walks read the stored spelling off
    os.walk and never follow links; the download route resolves, and without the root a link named
    unsloth-tmp (or a wrong-case entry on NTFS or APFS) would take the discount for a tree neither
    walk lists.
    """
    if not parts or parts[0] != _SANDBOX_TEMP_DIRNAME:
        return parts
    if root is not None and not _is_sandbox_temp_dir(
        os.path.join(root, _SANDBOX_TEMP_DIRNAME), root
    ):
        return parts
    return parts[1:]


_SERVABLE_SEGMENT_RE = re.compile(r"\A[^/\\\x00-\x1f]{1,255}\Z")


def _servable_segment(name: str) -> bool:
    if name in (".", "..") or not _SERVABLE_SEGMENT_RE.match(name):
        return False
    # Lone surrogates make encodeURIComponent throw.
    return not any("\ud800" <= ch <= "\udfff" for ch in name)


# Exact names we write, not a reserved pattern.
_INTERNAL_SANDBOX_FILES = frozenset({".unsloth_sandbox_remap.json", _SANDBOX_MARKER})


# Large files are identified by mtime and size alone.
_MAX_HASHED_SNAPSHOT_BYTES = 4 * 1024 * 1024


def _content_key(path: str, size: int) -> "str | None":
    """A digest for a file small enough to read, else None. On FAT/exFAT and some network volumes
    the timestamp granularity is a second or two, so an overwrite with different content of the
    same length inside one tick is invisible to mtime and size, and the call reported no file at
    all."""
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


_fine_mtime_devices: "set[int]" = set()


def _volume_timestamps_finely(workdir: str) -> bool:
    """Whether this volume records sub-second times, so digests can be skipped.

    That is the only thing the digest buys: at sub-second resolution a program cannot overwrite a
    file inside the tick of its own previous write, and reading every artifact twice per call was
    ~90% of a snapshot's cost.

    Read off the directory rather than probed with a file of our own: several chats can share a
    workdir, and a probe file appearing mid-walk is reported as a file the other call created. Three
    stamps because one whole-second reading on a fine volume is chance; three is not. Anything
    unreadable answers False, which hashes exactly as before.
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
    """Break a marker line the executed program printed itself. Both readers take the last one, so a
    call that created nothing and printed `__FILES__:[{"name": "report.csv", "size": 1}]` had
    that read as the envelope: the line was hidden from the model and the UI offered a download
    for a file nobody wrote. One space is enough, since both anchor on the line start, and it
    leaves the text otherwise as the program wrote it."""
    for marker in ("\n__FILES__:", "\n__IMAGES__:"):
        text = text.replace(marker, "\n " + marker[1:])
    return text


# Caps hashing per snapshot; the file cap alone allowed 2,000 x 4 MiB per walk.
_MAX_SNAPSHOT_HASH_BYTES = 64 * 1024 * 1024


def _snapshot_workdir_files(workdir: str | None) -> "dict[str, tuple]":
    """relative path -> change key for every regular file, for the post-run diff. All files, not
    just images: a .csv the model wrote used to be invisible. Size rides along with mtime because
    FAT/exFAT and some network volumes have coarse timestamps, where an overwrite inside one tick
    would look unchanged, and on those volumes alone a digest rides along too, since a
    same-length rewrite inside one tick matches on all of it otherwise."""
    snapshot: "dict[str, tuple]" = {}
    if not workdir or not os.path.isdir(workdir):
        return snapshot
    # Directories are budgeted separately: empty folders never reach the file cap.
    visited = 0
    hash_budget = 0 if _volume_timestamps_finely(workdir) else _MAX_SNAPSHOT_HASH_BYTES
    # walked to include nested outputs such as outputs/report.csv that a top-level listing drops.
    for base, dirs, names in os.walk(workdir):
        visited += 1
        if visited > _MAX_SNAPSHOT_DIRS:
            return snapshot
        relative = base[len(workdir) :].strip(os.sep)
        depth = len(_user_path_parts(relative.split(os.sep) if relative else []))
        # skip noisy dot directories except attachments; dot files like .gitignore remain valid artifacts.
        dirs[:] = (
            []
            if depth >= _MAX_SANDBOX_PATH_SEGMENTS - 1
            else [
                d
                for d in dirs
                if (not d.startswith(".") or (base == workdir and d == _ATTACHMENTS_DIR))
                and _servable_segment(d)
            ]
        )
        for name in names:
            # ignore internal markers only at the root; nested files with these names remain valid artifacts.
            if base == workdir and name in _INTERNAL_SANDBOX_FILES:
                continue
            if not _servable_segment(name):
                continue
            path = os.path.join(base, name)
            try:
                # one lstat replaces isfile, islink, and stat while rejecting the same non-regular entries.
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


# Scratch scripts of running calls; chats in a project share a workdir.
_active_scratch: "set[str]" = set()
_scratch_lock = threading.Lock()

# A call that ever overlapped another in this workdir claims nothing; no clock involved.
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
    """Whether a file changed between two snapshots of its directory. The digest only when both
    snapshots have one: hashing stops at a byte budget, so a file added or removed earlier in the
    walk can push an untouched later file in or out of it, and comparing the tuples whole would
    then report that file as one this call wrote."""
    if before[:2] != after[:2]:
        return True
    return before[2] is not None and after[2] is not None and before[2] != after[2]


def _created_file_sentinels(
    workdir: str | None,
    before: "dict[str, tuple]",
    exclude: "str | None" = None,
    token: "dict | None" = None,
) -> str:
    """Sentinels naming the files this call created or overwrote. ``__IMAGES__`` renders inline as
    before; ``__FILES__`` carries every file with its size so the UI can offer a download. Both
    are stripped before the model sees the result."""
    if token is not None and token.get("shared"):
        # Another call ran here meanwhile: a missing card beats one naming another chat's file.
        return ""
    after = _snapshot_workdir_files(workdir)
    if token is not None and token.get("shared"):
        return ""
    with _scratch_lock:
        scratch = set(_active_scratch)
    scratch.discard(exclude)
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

    images = [n for n in changed if os.path.splitext(n)[1].lower() in _IMAGE_EXTS]
    images = images[:_MAX_REPORTED_FILES]
    entries = []
    for name in changed[:_MAX_REPORTED_FILES]:
        try:
            size = os.stat(os.path.join(workdir, name)).st_size
        except OSError:
            size = None
        entries.append({"name": name, "size": size})

    # __IMAGES__ stays last: older clients slice from it to the end.
    out = f"\n__FILES__:{_json.dumps(entries)}"
    if images:
        out += f"\n__IMAGES__:{_json.dumps(images)}"
    return out


def _timed_out_result(
    output: str | None,
    timeout: int,
    workdir: str | None,
    scope: "str | None",
    omitted: "tuple[int, int]" = (0, 0),
) -> str:
    """Captured output, then the timeout status line.

    Output leads: a finished card shows the live stream when the result is a prefix of it
    (`preferFullToolOutput`), so a leading status would show the output twice.
    """
    ended = _truncate(f"Execution timed out after {timeout} seconds.")
    partial = _defuse_sentinels(output or "")
    if not partial.strip():
        return ended
    ctx = _window_context_tokens()
    head = _truncate(
        partial,
        workdir = workdir,
        scope = scope,
        reserve_tokens = _text_token_cost(f"\n{ended}", ctx),
        omitted = omitted,
    )
    result = f"{head}\n{ended}"
    room = _request_result_room()
    if room is not None and _text_token_cost(result, ctx) + _appended_by_the_loop(result) > room:
        return ended
    return result


def _python_exec(
    code: str,
    cancel_event = None,
    timeout: int = _EXEC_TIMEOUT,
    session_id: str | None = None,
    disable_sandbox: bool = False,
    output_callback = None,
    thread_id: str | None = None,
    *,
    tool_execution_mode: str = "auto",
    host_access_approved: bool = False,
) -> str:
    """Execute Python code in a subprocess sandbox. disable_sandbox (Bypass Permissions): skip the
    safety analysis and rlimit pre-exec, and use the host env minus secrets. output_callback:
    optional callable(str) streamed stdout as it is produced; the returned result is
    unchanged. tool_execution_mode selects automatic or required OS isolation; disable_sandbox
    keeps full access as a separate explicit choice."""
    if not code or not code.strip():
        return "No code provided."

    _guard_workdir = _tool_workdir_for_guard(session_id) if _needs_a_workdir(code) else None
    if _references_studio_credential_here(code, _guard_workdir) or _python_builds_a_credential_path(
        code, _guard_workdir
    ):
        return _STUDIO_CREDENTIAL_BLOCKED

    if not disable_sandbox:
        error = _check_code_safety(code)
        if error:
            # Capped: repeated forbidden constructs can produce an oversized error.
            return _truncate(error)
        # A same-UID child can read /proc/<ppid>/environ; best-effort since the child env is scrubbed.
        _harden_parent_against_proc_env_leak()
    elif not _harden_parent_against_proc_env_leak():
        # Fail closed if the /proc/<parent>/environ path cannot be closed.
        return (
            "Execution error: could not harden the Unsloth process against "
            "/proc environment reads; refusing bypass execution."
        )

    tmp_path = None
    _scratch_name = None
    # Bound before the try so the finally can release even when prepare raised.
    prepared = None
    try:
        workdir = _get_workdir(session_id)
        confinement = _account_confinement()
    except (ToolConfinementUnavailable, RetiredAccountError) as exc:
        return _truncate(f"Execution error: {exc}")
    # No spill in shared sandboxes (`_default` or a project): see `_spill_scope`.
    spill_scope = _spill_scope(session_id, thread_id)
    spill_dir = workdir if session_id else None
    call_token = _call_started(workdir)
    _before = _snapshot_workdir_files(workdir)
    try:
        # In the workdir: sys.path[0] keeps earlier helpers importable and __file__ in the sandbox.
        fd, tmp_path = tempfile.mkstemp(suffix = ".py", prefix = "studio_exec_", dir = workdir)
        # utf-8 so non-ASCII code survives Windows cp1252.
        _scratch_name = os.path.basename(tmp_path)
        with _scratch_lock:
            _active_scratch.add(_scratch_name)
        with os.fdopen(fd, "w", encoding = "utf-8") as f:
            f.write(code)

        safe_env = _build_bypass_env(workdir) if disable_sandbox else _build_safe_env(workdir)
        if disable_sandbox:
            safe_env = dict(safe_env)
            safe_env["PYTHONIOENCODING"] = "utf-8"
        popen_kwargs = dict(
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            # close_fds leaves 0-2 open, so an unset stdin would be the server's.
            stdin = subprocess.DEVNULL,
            text = True,
            encoding = "utf-8",
            errors = "replace",
        )
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        # -u streams output unbuffered without touching the child's os.environ.
        requested_mode = _requested_execution_mode(tool_execution_mode, disable_sandbox)
        base_preexec = (
            None
            if sys.platform == "win32"
            else (_bypass_preexec if disable_sandbox else _sandbox_preexec)
        )
        if base_preexec is _sandbox_preexec:
            _refresh_sandbox_memory_limit()
        # Managed accounts keep their own boundary as the outer contract; `confines`, not `is not None` (placeholder).
        if confinement is None or not confinement.confines:
            prepared = _prepare_tool_launch(
                os_sandbox.ToolLaunchPlan(
                    argv = (sys.executable, "-u", tmp_path),
                    workdir = workdir,
                    env = safe_env,
                    preexec_fn = base_preexec,
                    requested_mode = requested_mode,
                    timeout_seconds = timeout,
                    execution_kind = "python",
                    cancel_event = cancel_event,
                ),
                host_access_approved = host_access_approved and _reaches_host_paths("python", code),
            )
            proc = os_sandbox.spawn_prepared_launch(
                prepared, **_apply_prepared_launch(prepared, popen_kwargs)
            )
            _note_tool_execution(prepared.execution_record)
        else:
            popen_kwargs.update(cwd = workdir, env = safe_env)
            if sys.platform != "win32":
                popen_kwargs["preexec_fn"] = base_preexec
            argv = _apply_confinement(confinement, popen_kwargs, [sys.executable, "-u", tmp_path])
            proc = subprocess.Popen(argv, **popen_kwargs)

        pgid = _capture_process_group(proc)
        _adopt_tool_pid(proc.pid)

        if cancel_event is not None:
            watcher = threading.Thread(
                target = _cancel_watcher,
                args = (proc, cancel_event, 0.2, pgid),
                daemon = True,
            )
            watcher.start()

        # Always drain this way: it kills the group on cancel and keeps output byte-identical.
        output, timed_out, omitted = _drain_process_output(
            proc, timeout, output_callback, cancel_event, pgid = pgid
        )
        if prepared is not None:
            proc._unsloth_completion_reason = (
                "timed_out"
                if timed_out
                else (
                    "cancelled"
                    if cancel_event is not None and cancel_event.is_set()
                    else "finished"
                )
            )
        if prepared is not None:
            completion = os_sandbox.verify_prepared_completion(prepared, proc)
            if completion is not None and completion.get("timedOut"):
                timed_out = True
            _note_tool_execution(prepared.execution_record)
        # A run that wrote its file then hung still produced it.
        if timed_out:
            ended = _timed_out_result(output, timeout, spill_dir, spill_scope, omitted)
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
        # Detect on full output (truncation could hide the traceback); append after truncation.
        hint = _missing_path_hint(result, workdir)
        # Before the fit: defusing inserts characters, growing the result after measurement.
        result = _defuse_sentinels(result)
        result = (
            _truncate(result, workdir = spill_dir, scope = spill_scope, hint = hint, omitted = omitted)
            if result.strip()
            else "(no output)" + hint
        )

        # Without an id every first turn shares `_default`, so a card would serve another chat's file.
        if session_id:
            result += _created_file_sentinels(workdir, _before, _scratch_name, call_token)

        _forget_sandbox_capability_if_the_backend_failed(prepared, result)
        return result

    except os_sandbox.SandboxUnavailableError as e:
        if cancel_event is not None and cancel_event.is_set():
            # A stop during the (on Windows DACL, multi-second) probe is a cancel, not a sandbox error.
            return "Execution cancelled."
        return _sandbox_refusal(e)
    except Exception as e:
        if prepared is not None and prepared.backend == "mxc-processcontainer":
            _note_tool_execution(prepared.execution_record)
        return _truncate(f"Execution error: {e}")
    finally:
        _call_finished(call_token)
        if _scratch_name:
            with _scratch_lock:
                _active_scratch.discard(_scratch_name)
        if prepared is not None:
            prepared.cleanup()
            os_sandbox.finalize_prepared_cleanup(prepared)
            _note_tool_execution(prepared.execution_record)
        _forget_tool_pid(locals().get("proc"))
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


_CMD_MULTILINE_REFUSED = (
    "Execution error: the Terminal runs cmd, which runs only the first line of a multi-line "
    "command. Send one line per call and chain dependent commands with &&."
)


def _bash_exec(
    command: str,
    cancel_event = None,
    timeout: int = _EXEC_TIMEOUT,
    session_id: str | None = None,
    disable_sandbox: bool = False,
    output_callback = None,
    thread_id: str | None = None,
    *,
    tool_execution_mode: str = "auto",
    host_access_approved: bool = False,
) -> str:
    """Execute a bash command in a subprocess sandbox. disable_sandbox (Bypass Permissions): skip
    the command blocklist and rlimit pre-exec, and use the host env minus secrets.
    output_callback: optional callable(str) streamed stdout as it is produced; the returned
    result is unchanged. tool_execution_mode follows _python_exec."""
    if not command or not command.strip():
        return "No command provided."

    # Refused in every mode: this install's bearer would reach the serving provider.
    if _references_studio_credential_here(
        command, _tool_workdir_for_guard(session_id) if _needs_a_workdir(command) else None
    ):
        return _STUDIO_CREDENTIAL_BLOCKED

    # Chosen once, so the blocklist, env and argv all agree on the shell that will run this call.
    # Sandbox Low runs on the host shell: cmd is only picked to stay inside MXC.
    profile = _terminal_profile(disable_sandbox or tool_execution_mode == "software")
    if profile == "cmd_isolated":
        # Models often end a command with a newline; cmd /s /c cannot carry one.
        command = command.strip()
        if "\n" in command or "\r" in command:
            return _CMD_MULTILINE_REFUSED

    if not disable_sandbox:
        if profile in _CMD_PROFILES:
            # The cmd lexer misses separators glued to a word (a&powershell), cmd drops ^ escapes and
            # ' does not quote, so screen every reading.
            unescaped = command.replace("^", "")
            blocked = set().union(
                *(
                    _find_blocked_commands(text, posix = posix)
                    for text in (command, unescaped, _cmd_reading(command))
                    for posix in (False, True)
                )
            )
        else:
            blocked = _find_blocked_commands(command)
        if blocked:
            # Capped like the analyzer error.
            return _truncate(f"Blocked command(s) for safety: {', '.join(sorted(blocked))}")
        # A same-UID child can read /proc/<ppid>/environ; best-effort since the child env is scrubbed.
        _harden_parent_against_proc_env_leak()
    elif not _harden_parent_against_proc_env_leak():
        # Fail closed if the /proc/<parent>/environ path cannot be closed.
        return (
            "Execution error: could not harden the Unsloth process against "
            "/proc environment reads; refusing bypass execution."
        )

    workdir = None
    spill_dir = None
    spill_scope = None
    call_token = None
    prepared = None
    _scratch_name = None
    try:
        try:
            workdir = _get_workdir(session_id)
            confinement = _account_confinement()
        except (ToolConfinementUnavailable, RetiredAccountError) as exc:
            return _truncate(f"Execution error: {exc}")
        spill_scope = _spill_scope(session_id, thread_id)
        spill_dir = workdir if session_id else None
        call_token = _call_started(workdir)
        _before = _snapshot_workdir_files(workdir)
        safe_env = (
            _build_bypass_env(workdir)
            if disable_sandbox
            else _build_safe_env(workdir, shell = profile if profile == "cmd_isolated" else None)
        )
        popen_kwargs = dict(
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            stdin = subprocess.DEVNULL,
            text = True,
            # utf-8 with replace so the streaming reader never swallows a decode error.
            encoding = "utf-8",
            errors = "replace",
        )
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        if profile == "cmd_isolated":
            shell_argv, _scratch_name = [_windows_system_cmd(), "/c", command], None
        else:
            shell_argv, _scratch_name = _shell_argv(command, workdir, confinement)
        if _scratch_name:
            with _scratch_lock:
                _active_scratch.add(_scratch_name)
        requested_mode = _requested_execution_mode(tool_execution_mode, disable_sandbox)
        host_reach_approved = host_access_approved and _reaches_host_paths("terminal", command)
        if profile == "cmd_isolated" and requested_mode == "auto" and not host_reach_approved:
            # Screened only by the cmd lexer, so never replayed on the host if isolation drops.
            requested_mode = "required"
        base_preexec = (
            None
            if sys.platform == "win32"
            else (_bypass_preexec if disable_sandbox else _sandbox_preexec)
        )
        if base_preexec is _sandbox_preexec:
            _refresh_sandbox_memory_limit()
        if confinement is None or not confinement.confines:
            prepared = _prepare_tool_launch(
                os_sandbox.ToolLaunchPlan(
                    argv = tuple(shell_argv),
                    workdir = workdir,
                    env = safe_env,
                    preexec_fn = base_preexec,
                    requested_mode = requested_mode,
                    timeout_seconds = timeout,
                    execution_kind = "terminal",
                    cancel_event = cancel_event,
                ),
                host_access_approved = host_reach_approved,
            )
            proc = os_sandbox.spawn_prepared_launch(
                prepared, **_apply_prepared_launch(prepared, popen_kwargs)
            )
            _note_tool_execution(prepared.execution_record)
        else:
            popen_kwargs.update(cwd = workdir, env = safe_env)
            if sys.platform != "win32":
                popen_kwargs["preexec_fn"] = base_preexec
            argv = _apply_confinement(confinement, popen_kwargs, shell_argv)
            proc = subprocess.Popen(argv, **popen_kwargs)

        pgid = _capture_process_group(proc)
        _adopt_tool_pid(proc.pid)

        if cancel_event is not None:
            watcher = threading.Thread(
                target = _cancel_watcher,
                args = (proc, cancel_event, 0.2, pgid),
                daemon = True,
            )
            watcher.start()

        output, timed_out, omitted = _drain_process_output(
            proc, timeout, output_callback, cancel_event, pgid = pgid
        )
        if prepared is not None:
            proc._unsloth_completion_reason = (
                "timed_out"
                if timed_out
                else (
                    "cancelled"
                    if cancel_event is not None and cancel_event.is_set()
                    else "finished"
                )
            )
        if prepared is not None:
            completion = os_sandbox.verify_prepared_completion(prepared, proc)
            if completion is not None and completion.get("timedOut"):
                timed_out = True
            _note_tool_execution(prepared.execution_record)
        if timed_out:
            ended = _timed_out_result(output, timeout, spill_dir, spill_scope, omitted)
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
        hint = _missing_path_hint(result, workdir)
        result = _defuse_sentinels(result)  # before the fit; see _python_exec
        result = (
            _truncate(result, workdir = spill_dir, scope = spill_scope, hint = hint, omitted = omitted)
            if result.strip()
            else "(no output)" + hint
        )
        if session_id:
            result += _created_file_sentinels(workdir, _before, _scratch_name, call_token)
        _forget_sandbox_capability_if_the_backend_failed(prepared, result)
        return result

    except os_sandbox.SandboxUnavailableError as e:
        if cancel_event is not None and cancel_event.is_set():
            return "Execution cancelled."
        return _sandbox_refusal(e)
    except Exception as e:
        if prepared is not None and prepared.backend == "mxc-processcontainer":
            _note_tool_execution(prepared.execution_record)
        return _truncate(f"Execution error: {e}")
    finally:
        _call_finished(call_token)
        if prepared is not None:
            prepared.cleanup()
            os_sandbox.finalize_prepared_cleanup(prepared)
            _note_tool_execution(prepared.execution_record)
        _forget_tool_pid(locals().get("proc"))
        if _scratch_name:
            with _scratch_lock:
                _active_scratch.discard(_scratch_name)
            try:
                os.unlink(os.path.join(workdir, _scratch_name))
            except OSError:
                pass


# Imported at the end: the two modules reference each other.
from . import tool_path_approval as _path_gate  # noqa: E402

_path_gate._bind(globals())
_PATH_FLAG_SPECS = _path_gate._PATH_FLAG_SPECS
_python_reaches_outside_sandbox = _path_gate._python_reaches_outside_sandbox
_terminal_reaches_outside_sandbox = _path_gate._terminal_reaches_outside_sandbox
