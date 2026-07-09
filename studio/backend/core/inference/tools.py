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
                return _fold_cap(left * right)
            if isinstance(op, ast.Add):
                return _fold_cap(left + right)
            if isinstance(op, ast.Mod):
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
            return _fold_cap(fn(*args))
        except Exception:
            return None

    if isinstance(f, ast.Attribute):
        attr = f.attr
        owner = f.value
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
        "marshal.loads",
        "dill.loads",
        "cloudpickle.loads",
        "_pickle.loads",
        "jsonpickle.decode",
    }
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


def _build_exec_env(tree, const_env):
    """Map single-assignment names to exec builtins (`e = exec`) and to a compiled
    source (`c = compile("...")`) so a later call through the alias is unwrapped."""
    store_counts: dict[str, int] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            store_counts[n.id] = store_counts.get(n.id, 0) + 1

    exec_aliases: dict[str, str] = {}
    compiled_env: dict[str, tuple] = {}
    for stmt in getattr(tree, "body", []):
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            continue
        name = stmt.targets[0].id
        if store_counts.get(name, 0) != 1:
            continue
        rhs = stmt.value
        if isinstance(rhs, ast.Name) and rhs.id in _EXEC_BUILTINS:
            exec_aliases[name] = rhs.id
        elif (
            isinstance(rhs, ast.Call)
            and isinstance(rhs.func, ast.Name)
            and rhs.func.id == "compile"
            and rhs.args
        ):
            v = _const_fold(rhs.args[0], const_env)
            if isinstance(v, (str, bytes, bytearray)):
                compiled_env[name] = (_to_text(v), _compile_mode(rhs, const_env))
    return exec_aliases, compiled_env


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


def _recover_exec_payload(node, func_id, const_env, exec_aliases, compiled_env):
    """Recover a statically foldable source string for eval/exec/compile.

    Returns ("RECOVERED", src, mode) / ("DYNAMIC", None, None) / ("NO_PAYLOAD", None, None).
    """
    if not node.args:
        return ("NO_PAYLOAD", None, None)
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
            return ("RECOVERED", _to_text(v), _compile_mode(arg0, const_env))
        return ("DYNAMIC", None, None)

    # c = compile("..."); exec(c)
    if isinstance(arg0, ast.Name) and arg0.id in compiled_env:
        csrc, cmode = compiled_env[arg0.id]
        return ("RECOVERED", csrc, cmode)

    v = _const_fold(arg0, const_env)
    if isinstance(v, (str, bytes, bytearray)):
        mode = _compile_mode(node, const_env) if func_id == "compile" else base_mode
        return ("RECOVERED", _to_text(v), mode)
    return ("DYNAMIC", None, None)


# --------------------------------------------------------------------------
# Stage 3: first-class filesystem confinement.
#
# A destructive/mutating op is allowed only when its path is PROVABLY inside the
# session workdir (LOCAL). A read is blocked only when it PROVABLY escapes to a
# sensitive or traversal target (ESCAPE_READ). Path resolution is a constant-fold
# extended with os.path.join / pathlib join / f-string real-join + absolute-reset
# semantics; anything host-controlled or dynamic collapses to UNKNOWN.
# --------------------------------------------------------------------------
_PATH_DEPTH_CAP = 24
_PATHLIB_CTORS = frozenset(
    {"Path", "PurePath", "PosixPath", "WindowsPath", "PurePosixPath", "PureWindowsPath"}
)

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


def _classify_path_string(s):
    """LOCAL for a safe-relative path; ESCAPE for absolute / drive / ~ / `..`."""
    if isinstance(s, (bytes, bytearray)):
        s = _to_text(s)
    if not isinstance(s, str) or s == "":
        return "ESCAPE"  # empty path is not provably local -> fail closed
    norm = s.replace("\\", "/")
    if s[0] in ("/", "\\", "~"):
        return "ESCAPE"
    if len(s) >= 2 and s[1] == ":":
        return "ESCAPE"
    if ".." in norm.split("/"):
        return "ESCAPE"
    return "LOCAL"


def _is_pathlib_expr(node):
    """Whether an expression is structurally a pathlib.Path (ctor / join / attr chain)."""
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name) and f.id in _PATHLIB_CTORS:
            return True
        if isinstance(f, ast.Attribute):
            if f.attr in _PATHLIB_CTORS:
                return True
            if f.attr in (
                "joinpath",
                "with_name",
                "with_suffix",
                "absolute",
                "resolve",
                "expanduser",
                "parent",
            ) and _is_pathlib_expr(f.value):
                return True
        return False
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _is_pathlib_expr(node.left) or _is_pathlib_expr(node.right)
    if isinstance(node, ast.Attribute):
        return _is_pathlib_expr(node.value)
    return False


def _resolve_join(components, env, depth):
    """Combine component verdicts with join + absolute-reset semantics."""
    result = "LOCAL"
    for c in components:
        v = _resolve_path(c, env, depth + 1)
        if v == "ESCAPE":
            result = "ESCAPE"  # absolute reset or `..` -> outside
        elif v == "UNKNOWN" and result != "ESCAPE":
            result = "UNKNOWN"
    return result


def _resolve_path(
    node,
    env = None,
    depth = 0,
):
    """Classify a path expression as LOCAL / ESCAPE / UNKNOWN (see Stage 3)."""
    if node is None or depth > _PATH_DEPTH_CAP:
        return "UNKNOWN"

    v = _const_fold(node, env)
    if isinstance(v, (str, bytes, bytearray)):
        return _classify_path_string(v)

    if isinstance(node, ast.Name):
        rhs = (env or {}).get(node.id)
        if rhs is not None:
            return _resolve_path(rhs, env, depth + 1)
        return "UNKNOWN"

    if isinstance(node, ast.JoinedStr):
        # Not all-const (else it folded above): an absolute literal prefix escapes;
        # a relative prefix + dynamic hole cannot be proven local -> UNKNOWN.
        prefix = ""
        for part in node.values:
            if isinstance(part, ast.Constant):
                prefix += str(part.value)
            else:
                break
        if prefix:
            if prefix[0] in ("/", "\\", "~"):
                return "ESCAPE"
            if len(prefix) >= 2 and prefix[1] == ":":
                return "ESCAPE"
            if ".." in prefix.replace("\\", "/").split("/"):
                return "ESCAPE"
        return "UNKNOWN"

    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            left = _resolve_path(node.left, env, depth + 1)
            return "ESCAPE" if left == "ESCAPE" else "UNKNOWN"
        if isinstance(node.op, ast.Div):
            return _resolve_join([node.left, node.right], env, depth)
        return "UNKNOWN"

    if isinstance(node, ast.Call):
        return _resolve_path_call(node, env, depth)

    return "UNKNOWN"


def _resolve_path_call(node, env, depth):
    f = node.func
    attr = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")

    # os.path.join(...) / posixpath.join(...)
    if attr == "join" and isinstance(f, ast.Attribute):
        owner_fq = _fq_attr_name(f.value)
        if owner_fq.endswith("path") or owner_fq in ("op",):
            return _resolve_join(node.args, env, depth)
    if attr == "joinpath" and isinstance(f, ast.Attribute):
        return _resolve_join([f.value, *node.args], env, depth)
    # Host-controlled / absolute anchors are never provably local.
    if attr in (
        "expanduser",
        "expandvars",
        "abspath",
        "realpath",
        "getcwd",
        "getcwdb",
        "gettempdir",
        "mkdtemp",
        "home",
        "cwd",
    ):
        return "UNKNOWN"
    if attr == "normpath" and node.args:
        v = _const_fold(node.args[0], env)
        if isinstance(v, (str, bytes, bytearray)):
            return _classify_path_string(os.path.normpath(_to_text(v)))
        return "UNKNOWN"
    # Path(...) / PurePath(...) constructors (bare or pathlib.Path).
    if (isinstance(f, ast.Name) and f.id in _PATHLIB_CTORS) or (
        isinstance(f, ast.Attribute) and f.attr in _PATHLIB_CTORS
    ):
        if len(node.args) == 1:
            return _resolve_path(node.args[0], env, depth + 1)
        if len(node.args) >= 2:
            return _resolve_join(node.args, env, depth)
        return "UNKNOWN"
    return "UNKNOWN"


# Mutating-op inventory (fully-qualified stdlib names).
_FS_DELETE = frozenset(
    {
        "os.remove",
        "os.unlink",
        "os.rmdir",
        "os.removedirs",
        "shutil.rmtree",
        "pathlib.Path.unlink",
        "pathlib.Path.rmdir",
    }
)
_FS_META = frozenset(
    {"os.chmod", "os.lchmod", "os.chown", "os.lchown", "os.chflags", "os.truncate", "shutil.chown"}
)
_FS_MKDIR = frozenset({"os.mkdir", "os.makedirs", "os.mknod"})
_FS_CHDIR = frozenset({"os.chdir", "os.fchdir"})
_FS_SINGLE_MUTATE = _FS_DELETE | _FS_META | _FS_MKDIR | _FS_CHDIR
_FS_RENAME = frozenset({"os.rename", "os.renames", "os.replace", "shutil.move"})
_FS_COPY = frozenset(
    {
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
        "shutil.copytree",
        "shutil.copymode",
        "shutil.copystat",
    }
)
_FS_SYMLINK = frozenset({"os.symlink", "os.link"})
_FS_TEMPFILE = frozenset(
    {
        "tempfile.mkstemp",
        "tempfile.mkdtemp",
        "tempfile.NamedTemporaryFile",
        "tempfile.TemporaryFile",
        "tempfile.TemporaryDirectory",
        "tempfile.SpooledTemporaryFile",
    }
)
_FS_LIBWRITER_FQ = frozenset(
    {
        "numpy.save",
        "numpy.savez",
        "numpy.savez_compressed",
        "numpy.savetxt",
        "np.save",
        "np.savez",
        "np.savez_compressed",
        "np.savetxt",
        "torch.save",
        "joblib.dump",
        "cv2.imwrite",
    }
)
# Method-name-keyed library writers (receiver is a df / array / image / figure).
_FS_LIBWRITER_METHODS = frozenset(
    {"to_csv", "to_parquet", "to_pickle", "to_json", "to_excel", "to_feather", "savefig", "imwrite"}
)
# pathlib mutating methods -> (needs_receiver_path, extra_arg_index_or_None, op).
# unambiguous method names fire on any pathlib-looking receiver; the ambiguous
# ones (rename/replace/mkdir/chmod) require the receiver to be a pathlib expr.
_FS_PATHLIB_MUTATE = {
    "write_text": None,
    "write_bytes": None,
    "unlink": None,
    "rmdir": None,
    "symlink_to": 0,
    "hardlink_to": 0,
    "touch": None,
    "rename": 0,
    "replace": 0,
    "mkdir": None,
    "chmod": None,
}
_FS_PATHLIB_READ = frozenset({"read_text", "read_bytes"})


# --------------------------------------------------------------------------
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


def _build_shell_sink_aliases(tree):
    """Single-assignment names (stored exactly once) bound to a resolved shell sink."""
    os_aliases = {"os"}
    subprocess_aliases = {"subprocess"}
    from_aliases: dict[str, str] = {}
    store_counts: dict[str, int] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            store_counts[n.id] = store_counts.get(n.id, 0) + 1
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name == "os":
                    os_aliases.add(a.asname or "os")
                elif a.name == "subprocess":
                    subprocess_aliases.add(a.asname or "subprocess")
        elif isinstance(n, ast.ImportFrom) and n.module in ("os", "subprocess"):
            for a in n.names:
                fq = f"{n.module}.{a.name}"
                if fq in _SHELL_SINK_FUNCS:
                    from_aliases[a.asname or a.name] = fq

    aliases: dict[str, str] = {}
    for stmt in getattr(tree, "body", []):
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            continue
        name = stmt.targets[0].id
        if store_counts.get(name, 0) != 1:
            continue  # ambiguous reassignment -> do not alias (avoids FPs)
        fq = _resolve_static_shell_sink(stmt.value, os_aliases, subprocess_aliases, from_aliases)
        if fq:
            aliases[name] = fq
    return aliases


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
    if _budget is None:
        _budget = _AnalyzerBudget()
    if _analyzer_on:
        try:
            _const_env = _build_const_prop_env(tree)
            _exec_aliases, _compiled_env = _build_exec_env(tree, _const_env)
            _sink_aliases = _build_shell_sink_aliases(tree)
        except Exception:  # pragma: no cover - defensive: never crashier than legacy
            logger.warning("sandbox analyzer context build failed; legacy fallback", exc_info = True)
            _analyzer_on = False
            _const_env, _exec_aliases, _compiled_env = {}, {}, {}
            _sink_aliases = {}
    else:
        _const_env, _exec_aliases, _compiled_env = {}, {}, {}
        _sink_aliases = {}

    def _analyze_exec_call(node, func_id):
        """Stage 2 driver: recover + recurse a foldable payload, else dynamic policy."""
        try:
            kind, src, mode = _recover_exec_payload(
                node, func_id, _const_env, _exec_aliases, _compiled_env
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
    _GADGET_DUNDERS = frozenset(
        {
            "__subclasses__",
            "__bases__",
            "__base__",
            "__mro__",
            "__globals__",
            "__builtins__",
            "__code__",
            "__closure__",
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
                    return _sink_aliases.get(elt.id)
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
                # Stage 4: single-assignment alias `s = os.system; s('rm -rf /')`.
                if shell_func is None and _analyzer_on:
                    shell_func = _sink_aliases.get(func.id)
            elif _analyzer_on and isinstance(func, ast.Subscript):
                # Stage 4: inline literal container index `[os.system][0](...)`.
                shell_func = self._resolve_container_sink(func)

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
                elif _analyzer_on and func.id in _exec_aliases:
                    exec_func_id = _exec_aliases[func.id]

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
                is_dynamic_import = _ast_name_matches(func, _DYNAMIC_IMPORT_FUNCS) or (
                    isinstance(func, ast.Name) and func.id in ("__import__", "import_module")
                )
                # Deserialization sinks reconstruct arbitrary objects/code from bytes.
                if _analyzer_on and _fq_attr_name(func) in _CODE_DESERIALIZE_SINKS:
                    dynamic_desc = (
                        f"{_fq_attr_name(func)}() deserializes an unverifiable code payload"
                    )
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
                    if mod is None or mod.split(".")[0] in _DANGEROUS_IMPORT_NAMES:
                        dynamic_desc = "dynamic import of a computed or sensitive module name"
                elif (
                    isinstance(func, ast.Name)
                    and func.id in ("getattr", "setattr")
                    and node.args
                    and _ast_name_matches(
                        node.args[0],
                        _DYNAMIC_ATTR_TARGETS | self.os_aliases | self.subprocess_aliases,
                    )
                ):
                    # Stage 2 refinement: a benign constant attr (getattr(os, "getpid"))
                    # is allowed; only a dynamic attr or a dangerous constant attr blocks.
                    if _analyzer_on and len(node.args) >= 2:
                        attr_val = _const_fold(node.args[1], _const_env)
                        if isinstance(attr_val, str):
                            if attr_val in _DANGEROUS_ATTR_NAMES:
                                dynamic_desc = (
                                    f"{func.id}() on a sensitive module "
                                    "(attribute-name obfuscation)"
                                )
                        else:
                            dynamic_desc = (
                                f"{func.id}() on a sensitive module (attribute-name obfuscation)"
                            )
                    else:
                        dynamic_desc = (
                            f"{func.id}() on a sensitive module (attribute-name obfuscation)"
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

    _fs_read_strict = os.environ.get("FS_READ_STRICT", "0") != "0"

    def _fs_block(node, description):
        filesystem_violations.append(
            {
                "type": "filesystem_violation",
                "line": getattr(node, "lineno", -1),
                "description": description,
            }
        )

    def _fs_mutating(node, path_node, label):
        verdict = _resolve_path(path_node, _const_env)
        if verdict != "LOCAL":
            reason = (
                "escapes the session workdir"
                if verdict == "ESCAPE"
                else "cannot be proven to stay inside the session workdir"
            )
            _fs_block(
                node, f"{label}: destination path {reason} (must be a sandbox-local relative path)"
            )

    def _fs_libwriter(node, path_node, label):
        # Best-effort library writers block only on a PROVABLE escape; a dynamic
        # (UNKNOWN) path is left to the runtime realpath backstop to avoid
        # false-positiving on in-memory buffers.
        if _resolve_path(path_node, _const_env) == "ESCAPE":
            _fs_block(node, f"{label}: destination path escapes the session workdir")

    def _fs_read(node, path_node, label):
        v = _const_fold(path_node, _const_env)
        s = _to_text(v) if isinstance(v, (str, bytes, bytearray)) else None
        if s is not None:
            norm = s.replace("\\", "/")
            if s[:1] == "~" or ".." in norm.split("/"):
                _fs_block(node, f"{label}: read escapes the session workdir via traversal")
                return
            if _is_sensitive_abs_path(norm):
                _fs_block(node, f"{label}: reads a sensitive host identity / credential file")
                return
            return
        if _fs_read_strict and _resolve_path(path_node, _const_env) != "LOCAL":
            _fs_block(node, f"{label}: read path cannot be proven sandbox-local (FS_READ_STRICT)")

    def _kw(node, name):
        for kw in node.keywords or []:
            if kw.arg == name:
                return kw.value
        return None

    def _open_is_write(node):
        mode_node = node.args[1] if len(node.args) >= 2 else _kw(node, "mode")
        if mode_node is None:
            return False, "r"
        v = _const_fold(mode_node, _const_env)
        if isinstance(v, str):
            return any(c in v for c in "wax+"), v
        return True, None  # dynamic mode -> treat as write (conservative)

    class _FilesystemPolicyVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            fq = _fq_attr_name(node.func)
            f = node.func
            method = (
                f.attr
                if isinstance(f, ast.Attribute)
                else (f.id if isinstance(f, ast.Name) else "")
            )

            # Callee-independent literal-sensitive-path scan (library loaders that
            # internally open(): pandas.read_csv('/etc/shadow'), np.load('/etc/passwd')).
            for arg in list(node.args) + [kw.value for kw in (node.keywords or [])]:
                fv = _const_fold(arg, _const_env)
                sv = _to_text(fv) if isinstance(fv, (str, bytes, bytearray)) else None
                if sv is not None and _is_sensitive_abs_path(sv):
                    _fs_block(node, f"{sv!r} is a sensitive host identity / credential file")
                    break

            # builtins/io open(): write mode -> mutating; read mode -> read policy.
            is_open = (isinstance(f, ast.Name) and f.id == "open") or fq in ("io.open", "os.fdopen")
            if is_open and fq != "os.fdopen" and node.args:
                is_write, _mode = _open_is_write(node)
                if is_write:
                    _fs_mutating(node, node.args[0], "open(write)")
                else:
                    _fs_read(node, node.args[0], "open(read)")

            # os.open(path, flags): write flags -> mutating; else read.
            if fq == "os.open" and node.args:
                flags = node.args[1] if len(node.args) >= 2 else None
                flag_names = (
                    {n.attr for n in ast.walk(flags) if isinstance(n, ast.Attribute)}
                    if flags
                    else set()
                )
                is_write = (
                    flags is None
                    or bool(flag_names & {"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"})
                    or not flag_names
                )
                if is_write:
                    _fs_mutating(node, node.args[0], "os.open(write)")
                else:
                    _fs_read(node, node.args[0], "os.open(read)")

            if fq in _FS_SINGLE_MUTATE and node.args:
                _fs_mutating(node, node.args[0], fq)
            elif fq in _FS_RENAME and node.args:
                # rename/move: both src (removed) and dst are mutating.
                _fs_mutating(node, node.args[0], f"{fq} (source)")
                if len(node.args) >= 2:
                    _fs_mutating(node, node.args[1], f"{fq} (destination)")
                else:
                    dst = _kw(node, "dst")
                    if dst is not None:
                        _fs_mutating(node, dst, f"{fq} (destination)")
            elif fq in _FS_COPY and node.args:
                dst = node.args[1] if len(node.args) >= 2 else _kw(node, "dst")
                if dst is not None:
                    _fs_mutating(node, dst, f"{fq} (destination)")
                _fs_read(node, node.args[0], f"{fq} (source)")
            elif fq in _FS_SYMLINK and node.args:
                # os.symlink(src=target, dst=linkpath) / os.link: check BOTH.
                _fs_mutating(node, node.args[0], f"{fq} (target)")
                if len(node.args) >= 2:
                    _fs_mutating(node, node.args[1], f"{fq} (link path)")
            elif fq in _FS_TEMPFILE:
                d = _kw(node, "dir")
                if d is not None and _resolve_path(d, _const_env) != "LOCAL":
                    _fs_block(node, f"{fq}: dir= must be a sandbox-local relative path")
            elif fq in _FS_LIBWRITER_FQ and node.args:
                _fs_libwriter(node, node.args[0], fq)

            # Method-keyed library writers (df.to_csv(path), img.save(path), ...).
            if isinstance(f, ast.Attribute):
                if method in _FS_LIBWRITER_METHODS and node.args:
                    _fs_libwriter(node, node.args[0], method)
                elif method == "save" and node.args and fq not in _FS_LIBWRITER_FQ:
                    # PIL Image.save / model.save style: block only a provable escape.
                    _fs_libwriter(node, node.args[0], method)

            # pathlib mutating / reading methods on a Path-looking receiver. A plain
            # variable receiver is left to the Stage 5 runtime realpath backstop so
            # benign `p = Path("out.txt"); p.write_text(...)` is not over-blocked.
            if (
                isinstance(f, ast.Attribute)
                and _is_pathlib_expr(f.value)
                and (method in _FS_PATHLIB_MUTATE or method in _FS_PATHLIB_READ)
            ):
                recv = f.value
                if method in _FS_PATHLIB_READ:
                    _fs_read(node, recv, f"pathlib.Path.{method}")
                else:
                    _fs_mutating(node, recv, f"pathlib.Path.{method}")
                    extra = _FS_PATHLIB_MUTATE.get(method)
                    if extra is not None and len(node.args) > extra:
                        _fs_mutating(node, node.args[extra], f"pathlib.Path.{method} (target)")

            self.generic_visit(node)

    NetworkAndIoVisitor().visit(tree)

    if _analyzer_on:
        try:
            _FilesystemPolicyVisitor().visit(tree)
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
# The static gate is prove-or-block; this child-side guard resolves the true
# realpath (following symlinks) of every MUTATING file op and refuses it unless it
# lands inside the session workdir. It covers what static analysis cannot prove
# (dynamic paths, pre-existing symlinks, library writers that funnel through
# builtins.open). Reads are left unpatched. It is skipped entirely under
# disable_sandbox (Bypass Permissions).
# --------------------------------------------------------------------------
_SANDBOX_GUARD_SRC = r"""
import os as _os, builtins as _bi, functools as _ft
_WD = _os.path.realpath(__WORKDIR__)

def _within(p):
    try:
        if isinstance(p, int):
            return True
        rp = _os.path.realpath(_os.fspath(p))
    except Exception:
        return False
    return rp == _WD or rp.startswith(_WD + _os.sep)

def _deny(p, what):
    raise PermissionError(
        "sandbox: %s outside the session workdir is not permitted: %r" % (what, p)
    )

_real_open = _bi.open
@_ft.wraps(_real_open)
def _guarded_open(file, mode="r", *a, **k):
    m = mode if isinstance(mode, str) else "r"
    if any(c in m for c in "wax+") and not _within(file):
        _deny(file, "write")
    return _real_open(file, mode, *a, **k)
_bi.open = _guarded_open

def _wrap1(mod, name, what):
    orig = getattr(mod, name, None)
    if orig is None:
        return
    @_ft.wraps(orig)
    def w(path, *a, **k):
        if not _within(path):
            _deny(path, what)
        return orig(path, *a, **k)
    setattr(mod, name, w)

for _n in ("remove", "unlink", "rmdir", "removedirs", "truncate", "chmod",
           "chown", "mkdir", "makedirs"):
    _wrap1(_os, _n, _n)

def _wrap2(mod, name, both):
    orig = getattr(mod, name, None)
    if orig is None:
        return
    @_ft.wraps(orig)
    def w(src, dst, *a, **k):
        if both and not _within(src):
            _deny(src, name + " source")
        if not _within(dst):
            _deny(dst, name + " destination")
        return orig(src, dst, *a, **k)
    setattr(mod, name, w)

for _n in ("rename", "renames", "replace", "link", "symlink"):
    _wrap2(_os, _n, True)

try:
    import shutil as _sh
    _wrap1(_sh, "rmtree", "rmtree")
    _wrap2(_sh, "move", True)
    for _n in ("copy", "copy2", "copyfile", "copytree"):
        _wrap2(_sh, _n, False)
except Exception:
    pass

try:
    import pathlib as _pl
    def _wrapp(name, targ):
        orig = getattr(_pl.Path, name, None)
        if orig is None:
            return
        @_ft.wraps(orig)
        def w(self, *a, **k):
            if not _within(self):
                _deny(str(self), "Path." + name)
            if targ and a and not _within(a[0]):
                _deny(str(a[0]), "Path." + name + " target")
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
        file_body = code if disable_sandbox else (_sandbox_runtime_prelude(workdir) + code)
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
