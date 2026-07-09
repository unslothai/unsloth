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
    }
)
_BLOCKED_COMMANDS_COMMON = _BLOCKED_COMMANDS_COMMON | _INTERPRETER_COMMANDS | _CHILD_WRITE_COMMANDS
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
    if len(t) > 1 and t[-1] in "smhd":
        t = t[:-1]
    try:
        float(t)
        return True
    except ValueError:
        return False


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
    # position is detected even in `echo done; rm -rf x` (no whitespace) or
    # quote-split names (`r''m` collapses to `rm` after `;`).
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
    prev_was_flag = False  # previous token (while a wrapper is pending) was an option flag
    for token in tokens:
        if token in _SHELL_SEPARATORS or token in _SHELL_KEYWORDS_AS_SEP:
            expect_command = True
            prefix_pending = False
            prev_was_flag = False
            continue
        if token.startswith("-"):
            # Flags belong to the active command, but keep expect_command while a
            # wrapper prefix awaits its command (`stdbuf -oL cmd`, `xargs -- cmd`).
            if not prefix_pending:
                expect_command = False
            else:
                prev_was_flag = True
            continue
        if not expect_command:
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
        if base in _BLOCKED_COMMANDS:
            blocked.add(base)
        # Wrappers (env/time/xargs/sudo) consume one command; the next non-flag,
        # non-numeric token is the real command. sudo is also in _BLOCKED_COMMANDS.
        if base in _COMMAND_PREFIXES:
            prefix_pending = True
            continue
        expect_command = False
        prefix_pending = False

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
                blocked |= _find_blocked_commands(" ".join(seg))

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

    # A shell binary invoked with a SCRIPT FILE (`bash s.sh`) or `-s` (read the script from
    # stdin) runs unscanned shell code in the same unguarded environment; only the inline
    # `-c '...'` form is statically analyzable (handled above). Block a command-position
    # shell whose operands include a non-flag argument (the script) and no -c/-lc flag.
    _at_cmd_sh = True
    for i, tok in enumerate(tokens):
        if tok in _SHELL_SEPARATORS or tok in _SHELL_KEYWORDS_AS_SEP:
            _at_cmd_sh = True
            continue
        if _at_cmd_sh and os.path.basename(tok).lower() in _SHELLS:
            _has_c = False
            _script = None
            for k in range(i + 1, len(tokens)):
                t = tokens[k]
                if t in _SHELL_SEPARATORS or t in _SHELL_KEYWORDS_AS_SEP:
                    break
                tl = t.lower()
                if tl == "-c" or (
                    tl.startswith("-") and not tl.startswith("--") and tl.endswith("c")
                ):
                    _has_c = True
                    break
                if tl in ("-s", "--"):  # -s reads the script from stdin (unscanned)
                    _script = t
                    break
                if t.startswith("-"):
                    continue  # other shell flags: -l, -x, --login, --norc, ...
                _script = t  # first non-flag operand is the script file
                break
            if not _has_c and _script is not None:
                blocked.add("shell-script:" + _script)
            _at_cmd_sh = False
            continue
        if not tok.startswith("-"):
            _at_cmd_sh = False

    # Output redirection (> / >> / &> / N>) to a path OUTSIDE the workdir: a child shell
    # runs unguarded, so `echo x > /tmp/p` / `>> ../p` / `> ~/p` writes past the session
    # workdir. A relative literal target (> out.txt) stays in the workdir cwd and is
    # allowed; a NON-LITERAL target (variable / command substitution) cannot be verified,
    # so it fails closed (`echo x > "$p"` could expand anywhere). Scanning tokens (not the
    # raw string) avoids matching a `>` inside a quoted argument.
    for i, tok in enumerate(tokens):
        rm = re.search(r">{1,2}([^\s>]*)$", tok)
        if rm is None:
            continue
        tgt = rm.group(1)
        j = i
        # `>|` (noclobber override) and `>&` (stdout+stderr / fd-or-file redirect) tokenize
        # as `>` then `|` / `&`, so that punctuation is part of the redirect operator, not a
        # pipeline / background op; skip it and take the real target after. A pure fd target
        # (`>&2`) is a bare number that fails the path checks below and stays allowed.
        if not tgt and j + 1 < len(tokens) and tokens[j + 1] in ("|", "&"):
            j += 1
        if not tgt and j + 1 < len(tokens):
            tgt = tokens[j + 1]
        if not tgt:
            continue
        tn = tgt.replace("\\", "/")
        if (
            tgt.startswith("~")
            or tn.startswith("/")
            or ".." in tn.split("/")
            or "$" in tgt
            or "`" in tgt
        ):
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

    return blocked


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


class _FoldState:
    """Shared op counter + single-assignment const-prop environment."""

    __slots__ = ("ops", "names")

    def __init__(self, names = None):
        self.ops = 0
        self.names = names or {}


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

    # Count how many module-level stores each recorded name really has; if more
    # than one Store target references it anywhere, drop it.
    store_counts: dict[str, int] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            store_counts[n.id] = store_counts.get(n.id, 0) + 1

    env = {}
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
    }
)
# Modules whose load/loads/decode entry points run a pickle reduce payload; used to
# resolve `import pickle as p; p.loads(x)` and `from pickle import loads as l`.
_DESERIALIZE_MODULES = frozenset(
    {"pickle", "marshal", "dill", "cloudpickle", "_pickle", "jsonpickle"}
)
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
    new scope). Used so single-assignment alias detection is scope-correct."""
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
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                idx.enclosing[child] = func_enclose
                _rec(child, child, child)
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
                elif a.name == "subprocess":
                    subprocess_aliases.add(a.asname or "subprocess")
                elif a.name == "builtins":
                    builtins_aliases.add(a.asname or "builtins")
                elif a.name == "importlib":
                    importlib_aliases.add(a.asname or "importlib")
                if a.name in _DESERIALIZE_MODULES:
                    deser_module_aliases[a.asname or a.name] = a.name
        elif isinstance(n, ast.ImportFrom) and n.module in ("os", "subprocess"):
            for a in n.names:
                fq = f"{n.module}.{a.name}"
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
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    for scope in scopes:
        counts: dict[str, int] = {}
        rebound: set[str] = set()
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
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                rebound.update(n.names)
            elif (
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
            ):
                assigns.append((n.targets[0].id, n.value))
        idx.assigned[scope] = allnames
        smap: dict[str, str] = {}
        emap: dict[str, str] = {}
        cmap: dict[str, tuple] = {}
        camap: dict[str, bool] = {}
        imap: dict[str, bool] = {}
        dmap: dict[str, str] = {}
        scmap: dict[str, object] = {}
        rnmap: dict[str, ast.expr] = {}
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
            if fq:
                smap[name] = fq
            eb = _rhs_exec_builtin(rhs_eff)
            if eb is not None:
                emap[name] = eb
            elif _rhs_is_compile_call(rhs_eff) and rhs_eff.args:
                # Any `c = compile(...)` (bare / builtins.compile / from-import alias)
                # binds a code object, tracked for the types.FunctionType(c) execution
                # gadget below (dynamic or foldable payload).
                camap[name] = True
                v = _const_fold(rhs_eff.args[0], const_env)
                if isinstance(v, (str, bytes, bytearray)):
                    cmap[name] = (
                        _recovered_source(v),
                        _compile_mode(rhs_eff, const_env),
                        isinstance(v, (bytes, bytearray)),
                    )
            if _rhs_import_func(rhs_eff):
                imap[name] = True
            dfq = _rhs_deserializer(rhs_eff)
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
    if not node.args:
        return ("NO_PAYLOAD", None, None, False)
    arg0 = node.args[0]
    base_mode = "eval" if func_id == "eval" else "exec"

    # exec(compile("...", ...)) / eval(compile("...", "<s>", "eval"))
    if (
        isinstance(arg0, ast.Call)
        and isinstance(arg0.func, ast.Name)
        and arg0.func.id == "compile"
        and arg0.args
    ):
        v = _const_fold(arg0.args[0], const_env)
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
    if any(part in norm for part in _SANDBOX_SENSITIVE_DIR_PARTS):
        return True
    if _SANDBOX_SENSITIVE_RE.match(norm):
        return True
    low = norm.lower()
    return any(tok in low for tok in _SANDBOX_SENSITIVE_TOKENS)


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
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id in os_aliases:
            fq = f"os.{node.attr}"
            if fq in _SHELL_SINK_FUNCS:
                return fq
        if node.value.id in subprocess_aliases:
            fq = f"subprocess.{node.attr}"
            if fq in _SHELL_SINK_FUNCS:
                return fq
    if isinstance(node, ast.Name):
        return from_aliases.get(node.id)
    return None


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
            elif func_id != "compile":
                # An opaque, non-recoverable payload for an executing sink (eval/exec/
                # runpy) is a universal ACE bypass: it can synthesize any shell/network/
                # filesystem escape at runtime, invisibly to every static check. Block it
                # (compile() alone does not run, so it stays allowed -- the exec/eval of its
                # result is caught at that call). ast.literal_eval / json.loads cover data.
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
                elif alias.name == "inspect":
                    self.inspect_aliases.add(alias.asname or "inspect")
                elif alias.name == "operator":
                    self.operator_aliases.add(alias.asname or "operator")
                if alias.name in _DESERIALIZE_MODULES:
                    self.deserialize_module_aliases[alias.asname or alias.name] = alias.name
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
            elif node.module == "types":
                for alias in node.names:
                    if alias.name == "FunctionType":
                        self.functiontype_aliases.add(alias.asname or alias.name)
            elif node.module == "runpy":
                for alias in node.names:
                    if alias.name in ("run_path", "run_module"):
                        self.runpy_func_aliases.add(alias.asname or alias.name)
            elif node.module == "inspect":
                for alias in node.names:
                    if alias.name == "getclosurevars":
                        self.getclosurevars_aliases.add(alias.asname or alias.name)
            elif node.module == "operator":
                for alias in node.names:
                    if alias.name == "attrgetter":
                        self.attrgetter_aliases.add(alias.asname or alias.name)
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
            ci = _const_fold(sub.slice, _const_env)
            if isinstance(container, (ast.List, ast.Tuple)) and isinstance(ci, int):
                if -len(container.elts) <= ci < len(container.elts):
                    return _elt(container.elts[ci])
            if isinstance(container, ast.Dict) and ci is not None:
                for k, v in zip(container.keys, container.values):
                    if k is not None and _const_fold(k, _const_env) == ci:
                        return _elt(v)
            return None

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
            ci = _const_fold(sub.slice, _const_env)
            if isinstance(container, (ast.List, ast.Tuple)) and isinstance(ci, int):
                if -len(container.elts) <= ci < len(container.elts):
                    return _elt(container.elts[ci])
            if isinstance(container, ast.Dict) and ci is not None:
                for k, v in zip(container.keys, container.values):
                    if k is not None and _const_fold(k, _const_env) == ci:
                        return _elt(v)
            return None

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

        def _sink_ref_desc(self, n):
            """Describe ``n`` when it is a bare reference to a dangerous callable used as a
            first-class VALUE (map/reduce/partial argument): a dynamic-exec builtin, a shell
            sink (os.system / subprocess.*), a dynamic-import function, or a code
            deserializer. Returns a short description or None. The payloads such a sink runs
            never reach the recursive analyzer, so passing one by reference is unsafe."""
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

        def _is_compile_result(self, arg):
            """True when ``arg`` is a ``compile(...)`` code object (bare / builtins /
            single-assignment alias / inline-container unwrap). Used to catch code objects
            executed through ``types.FunctionType`` instead of eval/exec."""
            # (compile(src, ...),)[0] / [compile(...)][0] / {'k': compile(...)}['k']: a
            # trivial container unwrap hiding the compile() code object from the direct-call
            # check. Resolve the indexed element and recurse.
            if isinstance(arg, ast.Subscript):
                container = arg.value
                ci = _const_fold(arg.slice, _const_env)
                if isinstance(container, (ast.List, ast.Tuple)) and isinstance(ci, int):
                    if -len(container.elts) <= ci < len(container.elts):
                        return self._is_compile_result(container.elts[ci])
                if isinstance(container, ast.Dict) and ci is not None:
                    for k, v in zip(container.keys, container.values):
                        if k is not None and _const_fold(k, _const_env) == ci:
                            return self._is_compile_result(v)
                return False
            if isinstance(arg, ast.Call):
                af = arg.func
                if isinstance(af, ast.Name):
                    if af.id == "compile" or self.exec_from_aliases.get(af.id) == "compile":
                        return True
                    if _analyzer_on and _scope_idx.resolve(af.id, arg, "execb") == "compile":
                        return True
                elif (
                    isinstance(af, ast.Attribute)
                    and af.attr == "compile"
                    and _ast_name_matches(af.value, self.builtins_aliases)
                ):
                    return True
            if _analyzer_on and isinstance(arg, ast.Name):
                if _scope_idx.resolve(arg.id, arg, "compiledany"):
                    return True
            return False

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

        def visit_Call(self, node):
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

            # --- Dynamic execution / obfuscation primitives ---
            # eval / exec / compile (bare builtin or a single-assignment alias).
            exec_func_id = None
            if isinstance(func, ast.Name):
                if func.id in _DYNAMIC_EXEC_BUILTINS:
                    exec_func_id = func.id
                elif func.id in self.exec_from_aliases:
                    exec_func_id = self.exec_from_aliases[func.id]  # from builtins import exec as e
                elif _analyzer_on:
                    # single-assignment `e = exec` alias, resolved in the call's scope.
                    exec_func_id = _scope_idx.resolve(func.id, func, "execb")
            elif (
                isinstance(func, ast.Attribute)
                and func.attr in _DYNAMIC_EXEC_BUILTINS
                and _ast_name_matches(func.value, self.builtins_aliases)
            ):
                exec_func_id = func.attr  # builtins.eval(...) / __builtins__.exec(...)
            elif isinstance(func, ast.Attribute) and func.attr == "__call__":
                # eval.__call__("...") / exec.__call__(...) / builtins.eval.__call__(...)
                # invoke the builtin indirectly through its bound method; the payload is
                # still node.args[0], so recover + recurse it exactly like a direct call.
                _base = func.value
                if isinstance(_base, ast.Name):
                    if _base.id in _DYNAMIC_EXEC_BUILTINS:
                        exec_func_id = _base.id
                    elif _base.id in self.exec_from_aliases:
                        exec_func_id = self.exec_from_aliases[_base.id]
                    elif _analyzer_on:
                        exec_func_id = _scope_idx.resolve(_base.id, _base, "execb")
                elif (
                    isinstance(_base, ast.Attribute)
                    and _base.attr in _DYNAMIC_EXEC_BUILTINS
                    and _ast_name_matches(_base.value, self.builtins_aliases)
                ):
                    exec_func_id = _base.attr
            elif isinstance(func, ast.Subscript):
                # ({'e': exec}['e'])(...) / [exec][0](...): an inline container hides the
                # sink from the bare-name / attribute checks above.
                exec_func_id = self._resolve_container_exec(func)

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
                        # builtins.__import__('os') / __builtins__.__import__(...)
                        isinstance(_ecf, ast.Attribute)
                        and _ecf.attr == "__import__"
                        and _ast_name_matches(_ecf.value, self.builtins_aliases)
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
                    # sys.modules.get('os') -- the .get() twin of sys.modules['os'].
                    isinstance(func, ast.Attribute)
                    and func.attr == "get"
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "modules"
                    and _ast_name_matches(func.value.value, self.sys_aliases)
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
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "modules"
                    and _ast_name_matches(func.value.value, self.sys_aliases)
                ):
                    dynamic_desc = (
                        f"sys.modules.{func.attr}(...) mutates the loader table "
                        "(can drop a guarded module for reimport)"
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
                    )
                    and node.args
                    and self._is_compile_result(node.args[0])
                ):
                    dynamic_desc = (
                        "types.FunctionType() executes a compile() code object "
                        "(bypasses the eval/exec gate)"
                    )
                elif (
                    # runpy.run_path('evil.py') / runpy.run_module('evil') execute a
                    # file/module in the guarded interpreter WITHOUT the recursive source
                    # analysis exec/eval receive, so a sandboxed snippet can write a local
                    # evil.py and run it. Treat these as direct execution sinks. Covers the
                    # attribute form and a `from runpy import run_path` bare-name alias.
                    (
                        isinstance(func, ast.Attribute)
                        and func.attr in ("run_path", "run_module")
                        and _ast_name_matches(func.value, self.runpy_aliases)
                    )
                    or (isinstance(func, ast.Name) and func.id in self.runpy_func_aliases)
                ):
                    _rn = func.attr if isinstance(func, ast.Attribute) else func.id
                    dynamic_desc = f"runpy.{_rn}() executes a file/module without static analysis"
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
            self.generic_visit(node)

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
            # sys.modules['os'] pulls an already-loaded dangerous module out of the
            # loader table (os/subprocess are loaded by the host). Scope to a Load of a
            # dangerous LITERAL key so legit uses ("x" in sys.modules, sys.modules.get(
            # name), sys.modules[name] = ...) stay allowed.
            v = node.value
            # sys.modules[...] (attribute form) or getattr(sys, 'modules')[...] (the
            # getattr-obfuscated form) both index the loader table.
            is_sys_modules = (
                isinstance(v, ast.Attribute)
                and v.attr == "modules"
                and _ast_name_matches(v.value, self.sys_aliases)
            ) or (
                isinstance(v, ast.Call)
                and isinstance(v.func, ast.Name)
                and v.func.id == "getattr"
                and len(v.args) >= 2
                and _ast_name_matches(v.args[0], self.sys_aliases)
                and _extract_string_from_node(v.args[1]) == "modules"
            )
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
            # globals()['__builtins__'] / locals()[...] / vars()[...] pulls the builtins
            # namespace (or a dangerous module) out of the namespace dict, e.g.
            # getattr(globals()['__builtins__'], '__import__')('os'). Flag a Load of a
            # dangerous literal key off a bare globals()/locals()/vars() call.
            if isinstance(node.ctx, ast.Load) and (
                isinstance(v, ast.Call)
                and isinstance(v.func, ast.Name)
                and v.func.id in ("globals", "locals", "vars")
                and not v.args
            ):
                key = _const_fold(node.slice, _const_env)
                if isinstance(key, str) and (
                    key in ("__builtins__", "__builtin__")
                    or key.split(".")[0] in _DANGEROUS_IMPORT_NAMES
                ):
                    dynamic_exec.append(
                        {
                            "type": "dynamic_exec",
                            "line": getattr(node, "lineno", -1),
                            "description": "namespace-dict access to builtins / a sensitive module",
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
    # os/subprocess module aliases + from-import shell-name aliases, so a shell command
    # string that reads a host secret (os.system('cat /etc/passwd')) is scanned even when
    # os/subprocess is renamed.
    _os_mod_aliases = {"os"}
    _subprocess_mod_aliases = {"subprocess"}
    _shell_name_aliases: dict[str, str] = {}
    for _imp in ast.walk(tree):
        if isinstance(_imp, ast.ImportFrom) and _imp.module == "pathlib":
            for _a in _imp.names:
                if _a.name in _PATHLIB_CTORS:
                    _pathlib_ctor_aliases.add(_a.asname or _a.name)
        elif isinstance(_imp, ast.ImportFrom) and _imp.module in ("os", "io", "builtins"):
            for _a in _imp.names:
                if _a.name == "open":
                    _open_from_aliases.add(_a.asname or "open")
        elif isinstance(_imp, ast.ImportFrom) and _imp.module == "shutil":
            for _a in _imp.names:
                if _a.name in _SHUTIL_COPY_METHODS:
                    _shutil_copy_from_aliases.add(_a.asname or _a.name)
        elif isinstance(_imp, ast.ImportFrom) and _imp.module in ("os", "subprocess"):
            for _a in _imp.names:
                _fq = f"{_imp.module}.{_a.name}"
                if _fq in _SHELL_EXEC_FUNCS:
                    _shell_name_aliases[_a.asname or _a.name] = _fq
        elif isinstance(_imp, ast.Import):
            for _a in _imp.names:
                if _a.name == "shutil":
                    _shutil_aliases.add(_a.asname or "shutil")
                elif _a.name == "os":
                    _os_mod_aliases.add(_a.asname or "os")
                elif _a.name == "subprocess":
                    _subprocess_mod_aliases.add(_a.asname or "subprocess")

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
                and rhs.value.id in ("builtins", "__builtins__", "io", "os")
            ):
                return True
        if (
            isinstance(fn, ast.Attribute)
            and fn.attr == "open"
            and isinstance(fn.value, ast.Name)
            and fn.value.id in ("builtins", "__builtins__", "io", "os")
        ):
            return True
        return False

    def _is_shutil_copy_callee(fn):
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
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            if f.value.id in _os_mod_aliases:
                cand = f"os.{f.attr}"
            elif f.value.id in _subprocess_mod_aliases:
                cand = f"subprocess.{f.attr}"
            else:
                cand = None
            if cand in _SHELL_EXEC_FUNCS:
                return cand
        elif isinstance(f, ast.Name):
            return _shell_name_aliases.get(f.id)
        return None

    def _scan_shell_string_reads(node, f):
        # os.system('cat /etc/passwd') / subprocess.run('cat /etc/passwd', shell=True): the
        # read scanner otherwise treats the whole command as one opaque path candidate, and
        # _is_sensitive_abs_path ignores strings with whitespace. Tokenize the command and
        # check each token as a read path so an embedded host-secret read is caught.
        _fq = _shell_string_sink_fq(f)
        _is_str = _fq in _STRING_SHELL_SINKS
        if not _is_str:
            # subprocess.run/call/Popen/check_output/check_call(cmd, shell=True): a string
            # command with shell=True runs through /bin/sh (these are in _SHELL_EXEC_FUNCS
            # but not in _STRING_SHELL_SINKS, so check the shell= kwarg explicitly).
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                if f.value.id in _subprocess_mod_aliases and f.attr in (
                    "run",
                    "call",
                    "check_call",
                    "check_output",
                    "Popen",
                ):
                    for kw in node.keywords or []:
                        if kw.arg == "shell" and not (
                            isinstance(kw.value, ast.Constant) and kw.value.value is False
                        ):
                            _is_str = True
        if not _is_str or not node.args:
            return False
        cmd = _fold_read_arg(node.args[0])
        if cmd is None:
            return False
        try:
            toks = shlex.split(cmd, posix = True)
        except ValueError:
            toks = cmd.split()
        for t in toks:
            if t and not t.startswith("-") and _flag_read_path(node, t, True):
                return True
        return False

    class _SensitiveReadVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
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
            is_read_callee = (
                _resolves_to_open(f)
                or fq in ("io.open", "os.open")
                or fq in _SHUTIL_COPY_SINKS
                or _is_shutil_copy_callee(f)
                or (isinstance(f, ast.Name) and f.id in _shutil_copy_from_aliases)
                or _is_subprocess_exec_callee(f)
                or method in _READ_METHODS
            )
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
import os as _os, builtins as _bi, io as _io, pathlib as _pl
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
_WD = _realpath(__WORKDIR__)

def _within(p):
    try:
        if isinstance(p, int):
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

def _guard_open_like(real):
    @_gwraps(real)
    def w(file, mode="r", *a, **k):
        f = _fspath1(file)
        if _mode_is_write(mode) and not _within(f):
            _deny(f, "write")
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
        return real_open(path, flags, *a, **k)
    return _guarded
_os.open = _make_osopen_guard(_os.open)

def _wrap1(mod, name, what):
    orig = getattr(mod, name, None)
    if orig is None:
        return
    @_gwraps(orig)
    def w(path, *a, **k):
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
        return orig(p, *a, **k)
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
            if _mode_is_write(mode) and not _within(f):
                _deny(f, "FileIO write")
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
    # Path.open("w"): wrap the public method directly (mode-aware). Version-robust
    # because pathlib's accessor holds the original io.open (captured at the top).
    _real_path_open = _pl.Path.open
    @_gwraps(_real_path_open)
    def _guarded_path_open(self, mode="r", *a, **k):
        # Coerce mode through the base str (a str-subclass __contains__ must not lie).
        if _mode_is_write(mode) and not _within(self):
            _deny(str(self), "Path.open")
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
except Exception:
    pass
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
    turns any user program that opens with a future import into a SyntaxError. Parse
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
    has_future = False
    while (
        idx < len(body)
        and isinstance(body[idx], ast.ImportFrom)
        and body[idx].module == "__future__"
    ):
        has_future = True
        split = body[idx].end_lineno or split
        idx += 1
    if not has_future or split <= 0:
        return prelude + code
    lines = code.splitlines(keepends = True)
    head = "".join(lines[:split])
    tail = "".join(lines[split:])
    if head and not head.endswith(("\n", "\r")):
        head += "\n"
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

        proc = subprocess.Popen([sys.executable, tmp_path], **popen_kwargs)

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
