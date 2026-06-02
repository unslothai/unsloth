# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Tool definitions and executors for LLM tool calling.

Supports web search (DuckDuckGo), Python code execution, and terminal commands.
"""

import ast
import http.client
import os
import posixpath
import signal

os.environ["UNSLOTH_IS_PRESENT"] = "1"

import asyncio
import random
import re
import shlex
import ssl
import subprocess
import sys
import tempfile
import threading
import urllib.request

from core.inference.mcp_client import (
    MCP_TOOL_PREFIX,
    call_tool_sync,
    is_stdio,
    list_tools_async,
    parse_server_headers,
    probe_timeout,
    stdio_mcp_enabled,
)
from storage import mcp_servers_db

from loggers import get_logger

logger = get_logger(__name__)

_EXEC_TIMEOUT = 300  # 5 minutes

# Pre-import modules used in _sandbox_preexec at module level so that
# the preexec_fn closure does not trigger the import machinery in the
# forked child (which can deadlock in multi-threaded servers).
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

# Strict raster-image allowlist for sandbox file serving.
# No .svg (XSS risk via embedded scripts), no .html, no .pdf.
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


_SHELL_SEPARATORS = frozenset(
    {";", "&&", "||", "|", "&", "\n", "(", ")", "`", "{", "}"}
)
# Bash keywords that introduce a new command position (then $cmd, do $cmd, etc.).
_SHELL_KEYWORDS_AS_SEP = frozenset({"then", "do", "else", "elif"})
# Wrappers whose next non-flag argument is itself the command Bash will exec.
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


# Narrow allow-list of CLEAR credential / process-state targets.
#
# Two categories:
#
# * ``_HOME_RELATIVE_SENSITIVE`` — relative paths under the user's home that
#   are dangerous ONLY when accessed via a home-equivalent prefix (``~``,
#   ``$HOME``, ``${HOME}``, ``/home/<user>``, ``/Users/<user>``, ``/root``).
#   This is what keeps project-local files like ``./project/.npmrc`` /
#   ``./pkg/.pypirc`` readable while ``~/.npmrc`` is denied.
#
# * ``_ABSOLUTE_SENSITIVE`` — absolute paths that are dangerous wherever
#   they appear (`/etc/shadow`, `/proc/<pid>/environ`, etc.).
#
# Anything with a legitimate LLM-tool-use case (``~/.gitconfig``,
# ``~/.bashrc``, ``~/.ssh/config``, ``~/.ssh/known_hosts``, ``/etc/hosts``,
# ``~/.npm/`` cache, project-local rc files, ``~/.bash_history``,
# ``~/.cache/``) MUST stay out of this list — those still flow through.
# SSH private-key alternatives require a filename-end boundary so that
# the matching public key ``~/.ssh/id_rsa.pub`` (legitimate developer
# action) is NOT blocked. Non-key entries deliberately omit the end
# anchor: ``.aws/credentials.bak`` etc. are still credentials.
_SSH_KEY_END = r"(?=$|[\s'\";&|)<>])"
_HOME_RELATIVE_SENSITIVE = (
    # SSH private keys (config / known_hosts / *.pub intentionally allowed)
    rf"\.ssh/id_rsa{_SSH_KEY_END}",
    rf"\.ssh/id_ed25519{_SSH_KEY_END}",
    rf"\.ssh/id_ecdsa{_SSH_KEY_END}",
    rf"\.ssh/id_dsa{_SSH_KEY_END}",
    rf"\.ssh/identity{_SSH_KEY_END}",
    # Cloud provider credentials
    r"\.aws/credentials",
    r"\.docker/config\.json",
    r"\.kube/config",
    r"\.config/gcloud/application_default_credentials",
    r"\.config/gcloud/access_tokens",
    r"\.config/gcloud/credentials",
    # Personal package-manager tokens (project-local rc stays readable)
    r"\.pypirc",
    r"\.npmrc",
    r"\.cargo/credentials",
    # Authentication / password stores
    r"\.netrc",
    r"\.password-store",
    r"\.gnupg/private-keys-v1\.d",
)
_ABSOLUTE_SENSITIVE = (
    r"/etc/shadow",
    r"/etc/sudoers",
    r"/etc/ssh/ssh_host_[^\s'\"]+",
    # Linux process-state surfaces. ``thread-self`` and ``task/<tid>``
    # expose the same secrets as ``self``/``<pid>`` for individual
    # threads; ``cmdline`` and ``auxv`` carry env-derived strings too.
    r"/proc/(?:self|thread-self|\d+)/(?:environ|mem|maps|auxv|cmdline)",
    r"/proc/(?:self|thread-self|\d+)/task/\d+/(?:environ|mem|maps|auxv|cmdline)",
    # ``/proc/<pid>/cwd`` and ``/proc/<pid>/root`` are symlinks to the
    # process cwd and the filesystem root respectively. Reading via
    # ``/proc/self/cwd/X`` is equivalent to reading ``X`` but bypasses
    # any path normalisation that worked on the literal text; reading
    # ``/proc/self/root/etc/shadow`` opens ``/etc/shadow`` even under
    # chroot. Block any access via these symlink prefixes; there is no
    # legitimate LLM-tool-use reason to dereference them.
    r"/proc/(?:self|thread-self|\d+)/(?:cwd|root)(?:/|\Z)",
    r"/proc/(?:self|thread-self|\d+)/task/\d+/(?:cwd|root)(?:/|\Z)",
    r"/proc/kcore",
    r"/proc/kallsyms",
    r"/var/spool/cron/[^\s'\"]*",
)

# Home-equivalent prefix the path must be preceded by for HOME_RELATIVE
# entries to fire. Covers POSIX tilde forms (``~/`` and ``~user/``),
# $HOME / ${HOME}, POSIX absolute homes (/home/<u>, /root, /Users/<u>),
# and Windows env-var / drive-letter homes (%USERPROFILE%,
# %HOMEDRIVE%%HOMEPATH%, $env:USERPROFILE, C:/Users/<u>). Backslashes get
# normalized to forward slashes in _find_sensitive_paths before matching,
# so Windows-style C:\Users\... input is covered by the C:/Users/...
# branch here. ``~ubuntu/`` matches the POSIX ``~user/`` shell expansion
# that bash resolves to that user's home directory before exec.
_HOME_PREFIX_RE = (
    r"(?:"
    r"~(?:[^/\s'\";&|)<>]*)?"
    r"|\$\{?HOME\}?"
    r"|%USERPROFILE%"
    r"|%HOMEDRIVE%%HOMEPATH%"
    r"|\$env:USERPROFILE"
    r"|\$\{?env:USERPROFILE\}?"
    r"|/home/[^/\s'\"]+"
    r"|/root"
    r"|/Users/[^/\s'\"]+"
    r"|[A-Za-z]:/Users/[^/\s'\"]+"
    r")/"
)

# Path-token start anchor: refuse to match inside a longer path like
# ``./workspace/home/u/.aws/credentials`` or ``/tmp/home/u/.npmrc`` --
# those are project-local lookalikes, not host credentials. The negative
# lookbehind keeps matches anchored to a real shell token boundary.
_PATH_TOKEN_START = r"(?<![A-Za-z0-9_./~$%-])"

_HOME_SENSITIVE_RE = re.compile(
    _PATH_TOKEN_START
    + _HOME_PREFIX_RE
    + r"(?:"
    + "|".join(_HOME_RELATIVE_SENSITIVE)
    + r")",
    re.IGNORECASE,
)
_ABSOLUTE_SENSITIVE_RE = re.compile(
    _PATH_TOKEN_START + r"(?:" + "|".join(_ABSOLUTE_SENSITIVE) + r")",
    re.IGNORECASE,
)

# Whole-directory variants of the credential roots above. Only used by
# the shutil / file-copy gate -- ``ls ~/.ssh`` and ``find ~/.aws -type f``
# are legitimate, but ``shutil.copytree('~/.ssh', dst)`` and
# ``cp -r ~/.aws /tmp/out`` exfil every file in those dirs in one call.
#
# The end anchor matches the path AS the directory (``~/.ssh`` or
# ``~/.ssh/``) and not a file inside it (``~/.ssh/known_hosts`` —
# the per-file allow-list already governs whether that single read
# is OK). It also rejects similar-name prefixes (``~/.ssh_backup``).
_DIR_END = r"(?=/?$|/?[\s'\";&|)<>])"
_HOME_RELATIVE_SENSITIVE_DIRS = (
    rf"\.ssh{_DIR_END}",
    rf"\.aws{_DIR_END}",
    rf"\.config/gcloud{_DIR_END}",
    rf"\.gnupg{_DIR_END}",
    rf"\.docker{_DIR_END}",
    rf"\.kube{_DIR_END}",
    rf"\.password-store{_DIR_END}",
)
_ABSOLUTE_SENSITIVE_DIRS = (
    rf"/etc{_DIR_END}",
    rf"/etc/ssh{_DIR_END}",
    rf"/var/spool/cron{_DIR_END}",
    # Same Linux process-state roots as the per-file regex — copying
    # ``/proc/self/`` or ``/proc/<pid>/`` recursively drags the entire
    # process state (environ, mem, maps, cmdline) out.
    rf"/proc/(?:self|thread-self|\d+){_DIR_END}",
)
_HOME_SENSITIVE_DIR_RE = re.compile(
    _PATH_TOKEN_START
    + _HOME_PREFIX_RE
    + r"(?:"
    + "|".join(_HOME_RELATIVE_SENSITIVE_DIRS)
    + r")",
    re.IGNORECASE,
)
_ABSOLUTE_SENSITIVE_DIR_RE = re.compile(
    _PATH_TOKEN_START + r"(?:" + "|".join(_ABSOLUTE_SENSITIVE_DIRS) + r")",
    re.IGNORECASE,
)


def _matches_sensitive_dir(path: str) -> bool:
    """Return True if *path* names a sensitive credential / key directory
    (rather than a single file). Used by the shutil-copy gate so
    ``shutil.copytree('~/.ssh', dst)`` and ``shutil.copy('~/.aws', dst)``
    are caught even though ``~/.ssh`` itself isn't a single sensitive
    file in ``_HOME_RELATIVE_SENSITIVE``."""
    if not path:
        return False
    for cand in {path, path.replace("\\", "/")}:
        norm = _normalize_path_separators(cand)
        for projection in {cand, norm}:
            if _HOME_SENSITIVE_DIR_RE.search(projection):
                return True
            if _ABSOLUTE_SENSITIVE_DIR_RE.search(projection):
                return True
    return False


# Sensitive root prefix immediately followed by a shell substitution
# (``$(...)`` or backticks). Catches dynamic-path constructions like
# ``cat /etc/$(printf shadow)`` or ``cat /proc/1/$(echo environ)`` that
# materialise a protected path AFTER the literal scan has run.
_SENSITIVE_ROOT_WITH_EXPANSION_RE = re.compile(
    _PATH_TOKEN_START
    + r"(?:"
    + r"~(?:[^/\s'\";&|)<>]*)?/"
    + r"|\$\{?HOME\}?/"
    + r"|/home/[^/\s'\"]+/"
    + r"|/root/"
    + r"|/Users/[^/\s'\"]+/"
    + r"|/etc/"
    + r"|/proc/(?:self|thread-self|\d+)/"
    + r"|/var/spool/"
    + r")"
    + r"[^\s'\";&|`$]*"
    + r"(?:\$\([^)]*\)|`[^`]+`)",
    re.IGNORECASE,
)

# ``cp -r ~/.ssh /tmp/out`` / ``mv ~/.aws /tmp/out`` /
# ``tar czf out.tar.gz ~/.ssh`` -- bash directory-copy commands
# referencing a sensitive directory. The Python shutil gate covers
# the in-process equivalents (`shutil.copytree` etc.); without this
# pattern the bash side is asymmetric and `os.system('cp -r ~/.ssh
# /tmp/out')` slips through. The named commands cover the common
# dir-exfil verbs; ``rsync`` / ``zip`` / ``7z`` are added too because
# they all read the source directory recursively. ``ls`` / ``find``
# / ``cd`` / ``cat <single file>`` deliberately stay out of this
# list so legitimate inspection of sensitive directories is still
# allowed.
_BASH_DIR_EXFIL_COMMANDS = (
    "cp",
    "mv",
    "rsync",
    "tar",
    "zip",
    "7z",
    "7za",
    "xz",
    "scp",
    "sftp",
)
_BASH_SENSITIVE_DIR_NAMES = (
    r"\.ssh",
    r"\.aws",
    r"\.gnupg",
    r"\.kube",
    r"\.docker",
    r"\.config/gcloud",
    r"\.password-store",
)
_BASH_DIR_EXFIL_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(c) for c in _BASH_DIR_EXFIL_COMMANDS)
    + r")\b[^;&|\n]*?"
    + r"(?:"
    + _HOME_PREFIX_RE
    + r"(?:"
    + "|".join(_BASH_SENSITIVE_DIR_NAMES)
    + r")"
    + r"(?=/?$|/?[\s'\";&|)<>])"
    + r"|"
    + r"(?<![A-Za-z0-9_./~$%-])/etc(?=/?$|/?[\s'\";&|)<>])"
    + r"|"
    + r"(?<![A-Za-z0-9_./~$%-])/etc/ssh(?=/?$|/?[\s'\";&|)<>])"
    + r"|"
    + r"(?<![A-Za-z0-9_./~$%-])/var/spool/cron(?=/?$|/?[\s'\";&|)<>])"
    + r"|"
    + r"(?<![A-Za-z0-9_./~$%-])/proc/(?:self|thread-self|\d+)"
    + r"(?=/?$|/?[\s'\";&|)<>])"
    + r")",
    re.IGNORECASE,
)

# ``cat /etc/sha*ow`` / ``cat /etc/sh?dow`` -- bash expands ``*`` and
# ``?`` glob wildcards against the filesystem. The brace expander above
# only handles ``{a,b}`` braces; this pattern catches the wildcard
# globs that target a sensitive root path. The literal-text-only
# constraint (``[^\s'\";&|`$]*[*?]``) ensures we match an attached
# glob char and not a glob that lives in a separate argument like
# ``find /etc/ -name '*.conf'`` (whitespace breaks the token).
_SENSITIVE_ROOT_WITH_GLOB_RE = re.compile(
    _PATH_TOKEN_START
    + r"(?:"
    + r"~(?:[^/\s'\";&|)<>]*)?/"
    + r"|\$\{?HOME\}?/"
    + r"|/home/[^/\s'\"]+/"
    + r"|/root/"
    + r"|/Users/[^/\s'\"]+/"
    + r"|/etc/"
    + r"|/proc/(?:self|thread-self|\d+)/"
    + r"|/var/spool/"
    + r")"
    + r"[^\s'\";&|`$]*[*?]",
    re.IGNORECASE,
)

_BRACE_EXPANSION_RE = re.compile(r"\{([^{}]*,[^{}]*)\}")


_TILDE_USER_PREFIX_RE = re.compile(r"^~[^/]+/")


def _tail_escapes_home(tail: str) -> bool:
    """Return True if *tail* (the path after a home prefix) contains
    a ``..`` chain that takes the cursor above its starting directory.

    A simple ``startswith('..')`` check misses ``foo/../../etc/shadow``
    where a regular segment precedes the chain. Walks segments with a
    depth counter -- a negative depth at any point means the path has
    escaped its starting directory and the runtime resolve will land
    outside HOME (worst case ``/etc/shadow`` on a single-segment HOME
    like ``/root``)."""
    depth = 0
    for seg in tail.split("/"):
        if not seg or seg == ".":
            continue
        if seg == "..":
            depth -= 1
            if depth < 0:
                return True
        else:
            depth += 1
    return False


def _normalize_path_separators(text: str) -> str:
    """Collapse ``//`` to ``/``, remove ``/./`` segments, and resolve
    ``/..`` parent-directory traversal so that filesystem-equivalent
    spellings of a sensitive path (``/etc//shadow``, ``/etc/./shadow``,
    ``/etc/apt/../shadow``) match the canonical pattern.

    Home prefix handling. ``~/`` / ``$HOME/`` / ``${HOME}/`` /
    ``%USERPROFILE%/`` and POSIX ``~<user>/`` get re-attached after
    the parent-dir resolve so ``~/.ssh/../.aws/credentials`` becomes
    ``~/.aws/credentials``. When the ``..`` chain breaks out of HOME
    (``~/../etc/shadow``, ``~root/../etc/shadow``) the home prefix is
    DROPPED instead: with a single-segment sandbox HOME like ``/root``
    the runtime resolves ``~/../etc/shadow`` to ``/etc/shadow``, so
    the absolute projection has to reach ``_ABSOLUTE_SENSITIVE_RE``."""
    if not text:
        return text
    # Preserve the scheme separator (``http://``); collapse only path slashes.
    collapsed = re.sub(r"(?<!:)//+", "/", text)
    while "/./" in collapsed:
        collapsed = collapsed.replace("/./", "/")
    if collapsed.endswith("/."):
        collapsed = collapsed[:-2] or "/"
    if "/.." in collapsed or collapsed.endswith("/.."):
        # posixpath.normpath only follows ``..`` when the path is
        # absolute or starts with a known root. Re-attach the home
        # prefix unless ``..`` escaped home, in which case the
        # absolute form is what the runtime will hit.
        for prefix in ("~/", "$HOME/", "${HOME}/", "%USERPROFILE%/"):
            if collapsed.startswith(prefix):
                tail = collapsed[len(prefix) :]
                if _tail_escapes_home(tail):
                    return posixpath.normpath("/" + tail)
                return prefix + posixpath.normpath("/" + tail).lstrip("/")
        tilde_user = _TILDE_USER_PREFIX_RE.match(collapsed)
        if tilde_user:
            tail = collapsed[tilde_user.end() :]
            if _tail_escapes_home(tail):
                return posixpath.normpath("/" + tail)
            return tilde_user.group(0) + posixpath.normpath("/" + tail).lstrip("/")
        collapsed = posixpath.normpath(collapsed)
    return collapsed


def _expand_token_normalisations(token: str) -> set[str]:
    """Return the projections of a single token used for sensitive-path
    matching: raw, backslash-normalised, separator-collapsed."""
    out = {token}
    if "\\" in token:
        out.add(token.replace("\\", "/"))
    norm = _normalize_path_separators(token)
    if norm and norm != token:
        out.add(norm)
    return out


# Brace-defence sensitive names are SPLIT by root context so the gate
# does not over-block. ``cat ~/data/{maps,routes}`` is a legitimate
# user-data brace listing whose ``maps`` alternative is the name of
# a folder, NOT ``/proc/<pid>/maps``. Pairing each root with its own
# applicable sensitive-name set keeps the gate precise.

# Names that target a home / credential root. Apply to ``~/``,
# ``$HOME/``, ``/home/<u>/``, ``/root/``, ``/Users/<u>/``,
# ``%USERPROFILE%/`` -- the credential families that live under the
# user's home directory.
_HOME_BRACE_SENSITIVE_NAMES = (
    r"\.ssh/id_rsa",
    r"\.ssh/id_ed25519",
    r"\.ssh/id_ecdsa",
    r"\.ssh/id_dsa",
    r"\.aws/credentials",
    r"\.config/gcloud/[\w.]+",
    r"\.gnupg/[\w./-]+",
    r"\.netrc",
    r"\.pypirc",
    r"\.npmrc",
    r"\.docker/config\.json",
    r"\.kube/config",
)
# Names that target ``/etc/``: only the four well-defined credential /
# privilege files. ``hosts`` / ``hostname`` / ``resolv.conf`` /
# ``os-release`` are still allowed.
_ETC_BRACE_SENSITIVE_NAMES = (
    r"shadow",
    r"sudoers",
    r"passwd",
    r"gshadow",
)
# Names that target ``/proc/<pid>/``: the per-process state files that
# leak the runtime environment. Generic words like ``maps`` and
# ``mem`` only fire under this root, never under a home or local path.
_PROC_BRACE_SENSITIVE_NAMES = (
    r"environ",
    r"cmdline",
    r"maps",
    r"mem",
    r"auxv",
)


def _build_brace_re(prefix_alt: str, names: tuple[str, ...]) -> "re.Pattern[str]":
    """Compile a brace-aware sensitive-name regex for a single root
    alternation. Anchors:
      * ``_PATH_TOKEN_START`` -- shell-token boundary so project-local
        lookalikes (``./workspace/home/u/...``) do not match.
      * Path body between root and final brace can contain its own
        brace groups (the empty-alt + dummies bypass uses this).
      * ``(?<=[,{/])`` lookbehind plus ``(?=,|\\}|/)`` lookahead so
        the sensitive name is one complete brace alternative
        (``\\b`` does not fire between ``.`` and ``{`` -- both
        non-word -- so it cannot anchor here)."""
    return re.compile(
        _PATH_TOKEN_START
        + r"(?:"
        + prefix_alt
        + r")"
        + r"[^\s'\";&|`$]*?"
        + r"\{[^{}]*?(?<=[,{/])(?:"
        + "|".join(names)
        + r")(?=,|\}|/)[^{}]*\}",
        re.IGNORECASE,
    )


_HOME_BRACE_PREFIX_ALT = (
    r"~(?:[^/\s'\";&|)<>]*)?/+"
    + r"|\$\{?HOME\}?/+"
    + r"|/home/[^/\s'\"]+/+"
    + r"|/root/+"
    + r"|/Users/[^/\s'\"]+/+"
    + r"|%USERPROFILE%/+"
    + r"|%HOMEDRIVE%%HOMEPATH%/+"
)
_HOME_BRACE_RE = _build_brace_re(_HOME_BRACE_PREFIX_ALT, _HOME_BRACE_SENSITIVE_NAMES)
_ETC_BRACE_RE = _build_brace_re(r"/etc/+", _ETC_BRACE_SENSITIVE_NAMES)
_PROC_BRACE_RE = _build_brace_re(
    r"/proc/(?:self|thread-self|\d+)/+", _PROC_BRACE_SENSITIVE_NAMES
)
_VAR_SPOOL_BRACE_RE = _build_brace_re(r"/var/spool/cron/+", (r"[\w.-]+",))


def _expand_brace_projections(text: str, limit: int = 1024) -> set[str]:
    """Return the set of strings reachable from *text* by applying bash
    brace expansion ``{a,b}`` and bounded ``[abc]`` glob character
    classes. Bounded to ``limit`` total projections (raised from 64
    after a 64-alternative brace bomb -- ``cat ~/.aws/{x0,...,x62,
    credentials}`` -- evaded the per-alternative inner break, since the
    bypass adds 63 dummies plus the sensitive name in one brace group).

    Now expands ALL alternatives of the current brace in one inner
    pass so partially-applied state never blocks a sensitive name from
    being projected. The outer ``limit`` only stops the queue between
    brace groups, keeping the DOS bound while removing the off-by-one
    that capped the first brace at ``limit - 1`` alternatives."""
    out = {text}
    if "{" not in text and "[" not in text:
        return out
    queue = [text]
    glob_re = re.compile(r"\[([^\]/\\!^]{1,8})\]")
    while queue:
        if len(out) >= limit:
            break
        cur = queue.pop()
        brace = _BRACE_EXPANSION_RE.search(cur)
        if brace:
            for alt in brace.group(1).split(","):
                nxt = cur[: brace.start()] + alt + cur[brace.end() :]
                if nxt not in out:
                    out.add(nxt)
                    queue.append(nxt)
            continue
        klass = glob_re.search(cur)
        if klass:
            for ch in klass.group(1):
                if ch == "-":
                    continue
                nxt = cur[: klass.start()] + ch + cur[klass.end() :]
                if nxt not in out:
                    out.add(nxt)
                    queue.append(nxt)
    return out


def _find_sensitive_paths(command: str) -> set[str]:
    """Return any sensitive credential / process-state paths in *command*.

    Two-class matching:
      * Home-relative paths (``.ssh/id_rsa``, ``.aws/credentials``,
        ``.npmrc``, …) match only when prefixed by a home-equivalent
        token (``~/``, ``$HOME/``, ``/home/<user>/``, ``/root/``,
        ``/Users/<user>/``, ``%USERPROFILE%/``, ``C:/Users/<user>/``).
        This keeps project-local files like ``./project/.npmrc``
        readable.
      * Absolute system paths (``/etc/shadow``, ``/proc/<pid>/environ``,
        …) match anywhere they appear.

    To resist shell-quote splicing (``cat /etc/sha''dow``,
    ``cat ~/'.ssh/id_rsa'``) we scan three projections of the command:
    the raw text, a backslash-normalized copy (so Windows
    ``C:\\Users\\alice\\.ssh\\id_rsa`` is checked under the
    ``C:/Users/…`` branch), and a shlex-dequoted token reconstruction.
    Nested ``bash -c '…'`` / ``cmd /c '…'`` payloads are then recursed
    into so the bypass surface mirrors ``_find_blocked_commands``.

    Used by both ``_bash_exec`` (gates the raw command) and the Python
    AST gate (via ``_check_args_for_blocked``, so
    ``os.system('cat ~/.ssh/id_rsa')`` is caught the same way as the
    bash equivalent).

    The allow-list intentionally excludes common LLM-developer-tool
    paths (``~/.gitconfig``, ``~/.bashrc``, ``~/.ssh/config``,
    ``~/.ssh/known_hosts``, ``/etc/hosts``, ``~/.cache/``, ``*.pub``
    SSH public keys, project-local rc files) so legitimate tool calls
    like ``cat ~/.gitconfig`` or ``find src/ -name '*.py'`` still work.
    """
    if not command:
        return set()

    # Pre-normalise backslashes so the POSIX shlex below does not treat
    # ``C:\Users\alice`` as containing escape sequences (POSIX shlex
    # would otherwise collapse it to ``C:Usersalice`` and lose the path
    # structure). Both projections feed the regex scan.
    normalized = command.replace("\\", "/") if "\\" in command else command

    # Always use POSIX shlex for the dequote reconstruction regardless of
    # host OS: the threat model is shell-quote splicing (``cat /etc/sha''dow``,
    # ``bash -c "cat ~/'.ssh/id_rsa'"``) which is POSIX syntax. Running
    # non-POSIX shlex on Windows leaves the splice quotes intact and the
    # bypass slips through.
    try:
        lexer = shlex.shlex(normalized, posix = True, punctuation_chars = ";&|()`")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        tokens = normalized.split()

    raw_targets = [command]
    if normalized is not command:
        raw_targets.append(normalized)
    if tokens:
        raw_targets.append(" ".join(tokens))
        # Per-token normalisation catches ``..``-traversal that the
        # full-command normpath cannot resolve safely (commands aren't
        # paths). ``cat /etc/apt/../shadow`` reaches the regex as
        # ``/etc/shadow`` once the token is normalised in isolation.
        for tok in tokens:
            for variant in _expand_token_normalisations(tok):
                if variant != tok:
                    raw_targets.append(variant)

    # Cross-product the projections so the regexes see every shape:
    # raw / backslash-normalised / shlex-dequoted x with-and-without
    # path-separator normalisation x brace and glob expansions.
    scan_targets: set[str] = set()
    for text in raw_targets:
        for projected in _expand_brace_projections(text):
            scan_targets.add(projected)
            normalized_path = _normalize_path_separators(projected)
            if normalized_path != projected:
                scan_targets.add(normalized_path)

    found: set[str] = set()
    for text in scan_targets:
        for m in _HOME_SENSITIVE_RE.finditer(text):
            found.add(m.group(0))
        for m in _ABSOLUTE_SENSITIVE_RE.finditer(text):
            found.add(m.group(0))
        # Sensitive prefix + shell substitution that the literal scan
        # cannot statically resolve (``cat /etc/$(printf shadow)``).
        for m in _SENSITIVE_ROOT_WITH_EXPANSION_RE.finditer(text):
            found.add(m.group(0))
        # Sensitive prefix + bash glob (``cat /etc/sha*ow``,
        # ``cat /etc/sh?dow``, ``cat /etc/*``). The shell expands the
        # glob at runtime; statically we cannot enumerate the matches
        # but a glob immediately attached to a sensitive root is
        # an attempt to escape literal-path detection.
        for m in _SENSITIVE_ROOT_WITH_GLOB_RE.finditer(text):
            found.add(m.group(0))
        # Directory-copy verbs (``cp -r``, ``mv``, ``tar`` etc.) that
        # reference a sensitive directory. Asymmetry-fix for the
        # Python shutil dir-exfil gate that the round-4 commit added;
        # without this the bash side is still wide open.
        for m in _BASH_DIR_EXFIL_RE.finditer(text):
            found.add(m.group(0))
        # Brace-bomb defence. ``cat ~/{,x0,...,x341}/{.ssh/id_rsa,...}``
        # exceeds ``_expand_brace_projections``'s cap so the leaf
        # projection ``~/.ssh/id_rsa`` never reaches the literal regex.
        # These patterns catch sensitive-name fragments inside a brace
        # group attached to a sensitive root and fire regardless of
        # whether the expansion completed. Split by root so legitimate
        # brace listings like ``cat ~/data/{maps,routes}`` are not
        # flagged (``maps`` only matches under ``/proc/<pid>/``).
        for regex in (
            _HOME_BRACE_RE,
            _ETC_BRACE_RE,
            _PROC_BRACE_RE,
            _VAR_SPOOL_BRACE_RE,
        ):
            for m in regex.finditer(text):
                found.add(m.group(0))

    # Recurse into nested shells. Mirrors the structure in
    # _find_blocked_commands so ``bash -c "cat ~/.ssh/id_rsa"`` and
    # ``cmd /c type %USERPROFILE%\.aws\credentials`` both surface.
    _SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "csh", "tcsh", "fish"}
    _SHELLS_WIN = {"cmd", "cmd.exe"}
    for i, token in enumerate(tokens):
        tok_lower = token.lower()
        is_unix_c = tok_lower == "-c" or (
            tok_lower.startswith("-")
            and tok_lower.endswith("c")
            and not tok_lower.startswith("--")
        )
        is_win_c = tok_lower == "/c"
        if not (is_unix_c or is_win_c) or i < 1 or i + 1 >= len(tokens):
            continue
        for j in range(i - 1, -1, -1):
            prev = tokens[j]
            if prev.startswith("-"):
                continue
            if is_win_c and prev.startswith("/") and len(prev) <= 3:
                continue
            prev_base = os.path.basename(prev).lower()
            if is_unix_c and prev_base in _SHELLS:
                found |= _find_sensitive_paths(tokens[i + 1])
            elif is_win_c and prev_base in _SHELLS_WIN:
                found |= _find_sensitive_paths(tokens[i + 1])
            break
    return found


def _find_blocked_commands(command: str) -> set[str]:
    """Detect blocked commands at shell command position only.

    A token is at command position if it is the first token, or if the
    preceding token is a shell separator / brace-group opener / keyword
    that starts a new command (`then`, `do`, etc.), or a command-prefix
    wrapper like `env` / `time` / `xargs` (the next token is the real
    command). Tokens in argument position (`grep -r curl .`,
    `echo source the data`, `ls /usr/bin/curl`) are passed through.
    Also scans `find ... -exec CMD` and recurses into bash -c / cmd /c.
    """
    blocked: set[str] = set()

    # shlex with punctuation_chars splits `;`, `&&`, `||`, `|`, `(`, `)`, `` ` ``
    # off as their own tokens so we can detect command position even when a
    # caller writes `echo done; rm -rf x` (no whitespace) or quote-splits the
    # command name itself (`r''m` collapses to a single token `rm` at command
    # position after the `;` separator).
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
        # shlex may glue trailing meta-chars onto a token (`rm;`); strip them
        # so the basename match still hits `rm`. Leading shell-state chars
        # likewise.
        tok = tok.strip(";&|()`{}")
        base = os.path.basename(tok).lower()
        stem, ext = os.path.splitext(base)
        if ext in {".exe", ".com", ".bat", ".cmd"}:
            base = stem
        return base

    expect_command = True  # start of string is a command position
    prefix_pending = False  # last command-position token was env/time/timeout/xargs/...
    for token in tokens:
        if token in _SHELL_SEPARATORS or token in _SHELL_KEYWORDS_AS_SEP:
            expect_command = True
            prefix_pending = False
            continue
        if token.startswith("-"):
            # Flags belong to the active command. While a wrapper prefix is
            # waiting for its command (`stdbuf -oL cmd`, `xargs -- cmd`),
            # keep expect_command intact.
            if not prefix_pending:
                expect_command = False
            continue
        if not expect_command:
            continue
        # FOO=bar prefix: assignment list, next non-assignment token is the command.
        if _ASSIGNMENT_RE.match(token):
            continue
        # `timeout 1 cmd` / `nice -n 5 cmd` style numeric wrapper arg.
        if prefix_pending and token.lstrip("-").isdigit():
            continue
        base = _token_basename(token)
        if base in _BLOCKED_COMMANDS:
            blocked.add(base)
        # Wrappers (`env` / `time` / `xargs` / `sudo`) consume one command; the
        # next non-flag, non-numeric token is the real command. `sudo` is
        # already in _BLOCKED_COMMANDS, so it's flagged AND we keep walking.
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

    # Regex: blocked words at shell command boundaries that shlex won't see,
    # e.g. inside an unquoted $(rm -rf), <(rm), backtick chain, or appended to
    # a separator with no whitespace ("foo;rm"). Anchored to command-position
    # delimiters; does not match in argument position.
    lowered = command.lower()
    if _BLOCKED_COMMANDS:
        words_alt = "|".join(re.escape(w) for w in sorted(_BLOCKED_COMMANDS))
        pattern = (
            rf"(?:^|[;&|`\n(]\s*|[$]\(\s*|<\(\s*)"
            rf"(?:[\w./\\-]*/|[a-zA-Z]:[/\\][\w./\\-]*)?"
            rf"({words_alt})(?:\.(?:exe|com|bat|cmd))?\b"
        )
        blocked.update(re.findall(pattern, lowered))

    # Nested shell invocations (bash -c 'sudo whoami',
    #    bash -lc '...', bash --login -c '...', cmd /c '...').
    #    When a -c or /c flag is found, look backwards for a shell name
    #    (skipping intermediate flags like --login, -l, -x) and recursively
    #    scan the nested command string.
    _SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "csh", "tcsh", "fish"}
    _SHELLS_WIN = {"cmd", "cmd.exe"}
    for i, token in enumerate(tokens):
        tok_lower = token.lower()
        # Match -c exactly, or combined flags ending in c (e.g. -lc, -xc)
        is_unix_c = tok_lower == "-c" or (
            tok_lower.startswith("-")
            and tok_lower.endswith("c")
            and not tok_lower.startswith("--")
        )
        is_win_c = tok_lower == "/c"
        if not (is_unix_c or is_win_c) or i < 1 or i + 1 >= len(tokens):
            continue
        # Look backwards past any flags to find the shell binary.
        # On Unix, flags start with - (skip those). On Windows, flags
        # start with / but so do absolute paths, so only skip short
        # single-char /X flags (not /bin/bash style paths).
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

    Whitelist-built from scratch -- the parent process env is NOT inherited.
    Only PATH / HOME / TMPDIR / LANG / TERM / PYTHONIOENCODING (+ VIRTUAL_ENV
    or Windows SystemRoot when applicable) reach the child. HF_TOKEN,
    WANDB_API_KEY, AWS_*, GH_TOKEN, OPENAI_API_KEY, LD_PRELOAD, DYLD_*, and
    every other parent var are absent by construction. HOME points at the
    sandbox workdir so HF / wandb / aws SDKs cannot read cached credentials
    from the operator's real ~/.
    """
    # Start with the directory containing the running Python interpreter
    # so that subprocess calls to 'python', 'pip', etc. resolve to the
    # same environment the Studio server is running in.
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

    # Deduplicate while preserving order
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
    # Windows needs SystemRoot for Python/subprocess to work
    if sys.platform == "win32":
        env["SystemRoot"] = os.environ.get("SystemRoot", r"C:\Windows")
    return env


def _sandbox_preexec():
    """Best-effort sandbox setup for sandboxed subprocesses.

    Modules are resolved at import time so the forked child runs no imports.
    """
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

        # CLONE_NEWNET intentionally not applied: where userns is enabled it
        # blocks all egress, including allowlisted hosts. Network policy is
        # enforced by the AST host check and the bash blocklist.

    if _resource is not None:
        # RLIMIT_NPROC is per-real-UID, so the cap is well above normal usage.
        try:
            nproc = int(os.environ.get("UNSLOTH_STUDIO_SANDBOX_NPROC", "10000"))
            _resource.setrlimit(_resource.RLIMIT_NPROC, (nproc, nproc))
        except (ValueError, OSError, AttributeError):
            pass
        try:
            _resource.setrlimit(
                _resource.RLIMIT_FSIZE, (100 * 1024 * 1024, 100 * 1024 * 1024)
            )
        except (ValueError, OSError):
            pass
        try:
            as_bytes = (
                int(os.environ.get("UNSLOTH_STUDIO_SANDBOX_AS_GB", "8"))
                * 1024
                * 1024
                * 1024
            )
            _resource.setrlimit(_resource.RLIMIT_AS, (as_bytes, as_bytes))
        except (ValueError, OSError, AttributeError):
            pass
        try:
            cpu_s = int(os.environ.get("UNSLOTH_STUDIO_SANDBOX_CPU_S", "600"))
            _resource.setrlimit(_resource.RLIMIT_CPU, (cpu_s, cpu_s))
        except (ValueError, OSError, AttributeError):
            pass
        try:
            # Default high enough for multi-shard safetensors mmaps + Python's
            # own handle count; tunable via env for installs that hit the cap.
            # Clamp to the inherited hard limit so setrlimit doesn't ValueError
            # on machines where the parent's hard cap is below the requested
            # value (would otherwise leave NOFILE at the parent's default).
            nofile = int(os.environ.get("UNSLOTH_STUDIO_SANDBOX_NOFILE", "16384"))
            _soft_cur, hard_cur = _resource.getrlimit(_resource.RLIMIT_NOFILE)
            target = (
                nofile if hard_cur == _resource.RLIM_INFINITY else min(nofile, hard_cur)
            )
            _resource.setrlimit(_resource.RLIMIT_NOFILE, (target, target))
        except (ValueError, OSError, AttributeError):
            pass


def _get_shell_cmd(command: str) -> list[str]:
    """Return the platform-appropriate shell invocation for a command string."""
    if sys.platform == "win32":
        return ["cmd", "/c", command]
    return ["bash", "-c", command]


# Per-session working directories so each chat thread gets its own sandbox.
# Falls back to a shared ~/studio_sandbox/_default for API callers without a
# session_id.
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
        logger.warning(
            "Failed to resolve project sandbox for %s", session_id, exc_info = True
        )
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
            if not os.path.realpath(workdir).startswith(
                os.path.realpath(sandbox_root) + os.sep
            ):
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
            "Render a self-contained HTML/CSS/JavaScript artifact for the user. "
            "Call this at most once per assistant response unless the user "
            "explicitly asks for changes in that response. Future user requests "
            "for new artifacts may call render_html once. Put the entire document "
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
                    "description": "Short display title for the artifact.",
                },
            },
            "required": ["code"],
        },
    },
}

ALL_TOOLS = [WEB_SEARCH_TOOL, PYTHON_TOOL, TERMINAL_TOOL, RENDER_HTML_TOOL]


# OpenAI's function.name regex: ^[a-zA-Z0-9_-]{1,64}$ -- enforced before
# streaming starts. MCP servers can return tool names containing '.', '/',
# spaces, etc., which the prefix scheme would forward to OpenAI verbatim
# and 400 the whole request. Validate up front and skip with a warning.
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
        # OpenAI requires function.name ^[a-zA-Z0-9_-]{1,64}$; bad chars
        # (., /, spaces, etc.) or oversized names would 400 the whole
        # request. Skip + warn so the rest of the tools still ship.
        if not _OPENAI_FN_NAME_RE.fullmatch(name):
            logger.warning(
                "Skipping MCP tool '%s' on '%s': composed name '%s' is not "
                "valid OpenAI function.name (regex ^[a-zA-Z0-9_-]{1,64}$).",
                raw_name,
                display,
                name,
            )
            continue
        # Same MCP server returning duplicate tool names would also 400
        # OpenAI ("tools[N].function.name duplicates ..."). Drop dupes.
        if name in seen_names:
            logger.warning(
                "Skipping duplicate MCP tool '%s' on '%s'.", raw_name, display
            )
            continue
        seen_names.add(name)
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"[{display}] {tool.get('description') or ''}".strip(),
                    "parameters": tool.get("inputSchema")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return specs


async def get_enabled_mcp_tools() -> list[dict]:
    servers = [s for s in mcp_servers_db.list_servers() if s.get("is_enabled")]
    # Never spawn stdio servers when stdio is disabled on this host (e.g. a DB
    # carried over from a desktop install onto a Colab / network deployment).
    if not stdio_mcp_enabled():
        servers = [s for s in servers if not is_stdio(s["url"])]
    if not servers:
        return []

    results = await asyncio.gather(
        *(
            list_tools_async(
                url = s["url"],
                headers = parse_server_headers(s),
                timeout = probe_timeout(s["url"], bool(s.get("use_oauth"))),
                use_oauth = bool(s.get("use_oauth")),
            )
            for s in servers
        ),
        return_exceptions = True,
    )

    specs: list[dict] = []
    for server, payload in zip(servers, results):
        if isinstance(payload, BaseException):
            logger.warning(
                "MCP server '%s' (%s) discovery failed: %s",
                server.get("display_name") or server["id"],
                server.get("url"),
                payload,
            )
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
            f"Rendered HTML artifact: {safe_title}. Do not call render_html "
            "again in this response unless the user asks for changes. For a later "
            "user request for a new artifact, call render_html once."
        )
    return (
        "Rendered HTML artifact. Do not call render_html again in this response "
        "unless the user asks for changes. For a later user request for a new "
        "artifact, call render_html once."
    )


def execute_tool(
    name: str,
    arguments: dict,
    cancel_event = None,
    timeout: int | None = _TIMEOUT_UNSET,
    session_id: str | None = None,
) -> str:
    """Execute a tool by name with the given arguments. Returns result as a string.

    ``timeout``: int sets per-call limit in seconds, ``None`` means no limit,
    unset (default) uses ``_EXEC_TIMEOUT`` (300 s).
    ``session_id``: optional thread/session ID for per-conversation sandbox isolation.
    """
    logger.info(
        f"execute_tool: name={name}, session_id={session_id}, timeout={timeout}"
    )
    effective_timeout = _EXEC_TIMEOUT if timeout is _TIMEOUT_UNSET else timeout
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
            arguments.get("code", ""), cancel_event, effective_timeout, session_id
        )
    if name == "terminal":
        return _bash_exec(
            arguments.get("command", ""), cancel_event, effective_timeout, session_id
        )
    return f"Unknown tool: {name}"


_MAX_PAGE_CHARS = 16000  # limit fetched page text (after HTML-to-MD conversion)
# Raw download cap.  Must be larger than _MAX_PAGE_CHARS because SSR pages
# embed large <head> sections (CSS, JS, SVGs) that are stripped during
# HTML-to-Markdown conversion.  512 KB is enough to reach article content
# on GitBook / Next.js / Docusaurus pages whose <head> alone can be 200 KB.
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
    """HTTPS connection that connects to a pinned IP but uses a different
    hostname for SNI and certificate verification.

    The SSRF IP-pinning rewrites URLs to raw IPs.  A normal HTTPSConnection
    would then send no SNI and verify the cert against the IP, both of which
    fail.  This subclass splits the two concerns: TCP connects to the pinned
    IP (``host`` parameter) while TLS uses ``sni_hostname`` for the
    ClientHello and cert check.
    """

    def __init__(self, host: str, *, sni_hostname: str, **kwargs):
        super().__init__(host, **kwargs)
        self._sni_hostname = sni_hostname

    def connect(self):
        # TCP connect to the pinned IP stored in self.host (+ tunnel if
        # a proxy is configured via set_tunnel, though we do not use one).
        http.client.HTTPConnection.connect(self)
        # TLS handshake with the real hostname for SNI + cert verification.
        self.sock = self._context.wrap_socket(
            self.sock,
            server_hostname = self._sni_hostname,
        )


class _SNIHTTPSHandler(urllib.request.HTTPSHandler):
    """HTTPS handler that sends the correct SNI hostname during TLS handshake.

    The SSRF IP-pinning rewrites URLs to raw IPs, which breaks SNI and cert
    verification.  This handler returns a ``_PinnedHTTPSConnection`` that
    connects to the pinned IP but verifies TLS against the original hostname.
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

    Returns ``(ok, reason_or_empty, resolved_ip)``.  The caller should
    connect to *resolved_ip* (with a ``Host`` header) to prevent DNS
    rebinding between validation and the actual fetch.
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
        # `not ip.is_global` rejects every category the denylist below
        # also rejects PLUS shared address space (100.64.0.0/10 carrier-
        # grade NAT) and benchmarking/documentation/exchange ranges that
        # Python classifies with `is_private=False` and `is_global=False`
        # (see https://docs.python.org/3/library/ipaddress.html#ipaddress.IPv4Address.is_global).
        # The explicit predicates after it give human-readable categories
        # in the error message, but a single non-global check is the
        # source of truth and prevents future ranges from leaking.
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

    # Return the first resolved address for pinning
    first_ip = infos[0][4][0]
    return True, "", first_ip


def _fetch_page_text(
    url: str, max_chars: int = _MAX_PAGE_CHARS, timeout: int = 30
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
            # Pin to the validated IP to prevent DNS rebinding.
            # Rewrite the URL to use the IP and set the Host header.
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
                    return (
                        f"Failed to fetch URL: HTTP {e.code} {getattr(e, 'reason', '')}"
                    )
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
            # Success -- read capped body
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

    # Convert HTML to Markdown using the builtin converter (no external deps)
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
    # Direct URL fetch mode
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


def _check_signal_escape_patterns(code: str):
    """
    Check if code contains patterns that could escape signal-based timeouts.

    Vendored from unsloth_zoo.rl_environments to avoid importing unsloth_zoo
    (which requires GPU drivers and fails on Mac/Apple Silicon).

    Returns (safe: bool, details: dict)
    """
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

    # Dangerous os/subprocess functions that can execute shell commands
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

    # Simple ``name = 'literal'`` assignments are tracked on a pre-pass
    # below and stored here so ``_extract_string_from_node`` can fold
    # them as if they were inline string constants. Same surface for
    # function aliases (``e = eval``) populates ``eval_exec_aliases``.
    #
    # ``string_bindings`` returns a single representative string per
    # name (used by callers via ``_extract_string_from_node``).
    # ``string_bindings_all`` keeps EVERY literal value ever bound to
    # a name; the representative is picked to favour sensitive-shaped
    # paths so an adversarial ``p = '/tmp/safe'; p = '/etc/shadow';
    # open(p)`` (Python last-wins at runtime) does not slip through
    # the gate just because the AST walk picked the safe binding first.
    string_bindings: dict[str, str] = {}
    string_bindings_all: dict[str, list[str]] = {}
    eval_exec_aliases: dict[str, str] = {}

    # ``os.path.join`` alias tracking. Recognised forms:
    #
    #   import os                       -> "os.path.join"
    #   import os as o                  -> "o.path.join"
    #   from os import path             -> "path.join"
    #   from os import path as op       -> "op.join"
    #   import posixpath / ntpath / as pp -> "pp.join"
    #   from os.path import join        -> bare "join(...)"
    #   from os.path import join as j   -> bare "j(...)"
    #   from posixpath import join      -> bare "join(...)"
    #
    # ``os_path_module_aliases`` holds the dotted prefix used for an
    # attribute call (``o``, ``op``, ``pp``, ...) such that
    # ``<alias>.join(...)`` is treated as ``os.path.join``.
    # ``bare_path_join_aliases`` holds bare-name callables that
    # behave like ``os.path.join`` when called directly.
    os_path_module_aliases: set[str] = {"os.path", "posixpath", "ntpath"}
    bare_path_join_aliases: set[str] = set()
    bare_path_expanduser_aliases: set[str] = set()

    # ``shutil`` alias tracking. Recognised forms:
    #
    #   import shutil                       -> "shutil"
    #   import shutil as sh                 -> "sh"
    #   from shutil import copyfile         -> bare "copyfile(...)"
    #   from shutil import copy as cp       -> bare "cp(...)"
    shutil_module_aliases: set[str] = {"shutil"}
    bare_shutil_copy_aliases: dict[str, str] = {}

    _SHUTIL_COPY_NAMES = (
        "copyfile",
        "copy",
        "copy2",
        "copytree",
        "move",
    )

    # ``pathlib`` alias tracking for the pre-pass pathlib resolver.
    # Visitor-level state extends these later, but the pre-pass needs
    # them now so ``import pathlib as pl; p = pl.Path('/etc/shadow')``
    # is folded into ``string_bindings``. Mirror of ``_PATHLIB_PATH_CLASSES``
    # below; kept literal here to avoid a forward-reference dance.
    _PATHLIB_PATH_CLASSES_PREPASS = (
        "Path",
        "PurePath",
        "PosixPath",
        "WindowsPath",
        "PurePosixPath",
        "PureWindowsPath",
    )
    pathlib_module_aliases_prepass: set[str] = {"pathlib"}
    path_class_aliases_prepass: set[str] = set(_PATHLIB_PATH_CLASSES_PREPASS)

    def _run_alias_prepass(subtree: ast.AST) -> None:
        """Collect import aliases (os/os.path/posixpath/shutil/pathlib)
        from ``subtree``. Idempotent and additive so eval/exec payloads
        that contain ``import shutil as sh`` see their aliases tracked
        before the inner visitor runs."""
        for _node in ast.walk(subtree):
            if isinstance(_node, ast.Import):
                for alias in _node.names:
                    _local = alias.asname or alias.name
                    if alias.name == "os":
                        os_path_module_aliases.add(f"{_local}.path")
                    elif alias.name in ("posixpath", "ntpath"):
                        os_path_module_aliases.add(_local)
                    elif alias.name == "shutil":
                        shutil_module_aliases.add(_local)
                    elif alias.name == "pathlib":
                        pathlib_module_aliases_prepass.add(_local)
            elif isinstance(_node, ast.ImportFrom):
                if _node.module == "os":
                    for alias in _node.names:
                        if alias.name == "path":
                            os_path_module_aliases.add(alias.asname or "path")
                elif _node.module == "os.path" or _node.module in (
                    "posixpath",
                    "ntpath",
                ):
                    for alias in _node.names:
                        if alias.name == "join":
                            bare_path_join_aliases.add(alias.asname or "join")
                        elif alias.name == "expanduser":
                            bare_path_expanduser_aliases.add(
                                alias.asname or "expanduser"
                            )
                elif _node.module == "shutil":
                    for alias in _node.names:
                        if alias.name in _SHUTIL_COPY_NAMES:
                            bare_shutil_copy_aliases[alias.asname or alias.name] = (
                                f"shutil.{alias.name}"
                            )
                elif _node.module == "pathlib":
                    for alias in _node.names:
                        if alias.name in _PATHLIB_PATH_CLASSES_PREPASS:
                            path_class_aliases_prepass.add(alias.asname or alias.name)

    _run_alias_prepass(tree)

    # ``_SENSITIVE_FILE_PREFIXES`` and ``_SENSITIVE_FILE_RE`` are also
    # defined inside ``NetworkAndIoVisitor`` for the open-call gate,
    # but ``_looks_sensitive`` needs them in the binding pre-pass which
    # runs much earlier. Duplicate the literal here so the bias check
    # covers ``/etc/passwd`` (not in ``_ABSOLUTE_SENSITIVE``, only in
    # this prefix list) too.
    _PREPASS_SENSITIVE_PREFIXES = (
        "/etc/passwd",
        "/etc/shadow",
        "/etc/sudoers",
        "/etc/ssh/",
    )
    _PREPASS_SENSITIVE_RE = re.compile(
        r"^/proc/(?:self|\d+)/(?:environ|cmdline|task/\d+/environ)$"
    )

    def _looks_sensitive(value: str) -> bool:
        """True if *value* matches any host-credential / process-state
        path that the bash / file gates already flag. Uses the
        authoritative ``_find_sensitive_paths`` (covers /etc/shadow,
        /proc/<pid>/environ, ~/.ssh/id_rsa, ~/.aws/credentials, etc.)
        plus the open-call ``_SENSITIVE_FILE_PREFIXES`` / ``_SENSITIVE_FILE_RE``
        so /etc/passwd and similar prefix-only entries are caught too."""
        if not value:
            return False
        if _find_sensitive_paths(value):
            return True
        if any(value.startswith(p) for p in _PREPASS_SENSITIVE_PREFIXES):
            return True
        if _PREPASS_SENSITIVE_RE.match(value):
            return True
        return False

    def _record_string_binding(name: str, value: str) -> None:
        """Append ``value`` to ``string_bindings_all[name]`` and update
        ``string_bindings[name]`` so the gate sees the most sensitive
        value the variable could carry at runtime. The selection rule
        mirrors Python's last-wins semantics for sensitive values:

          * If the new value is sensitive, it always wins (even if the
            current is also sensitive) -- a later sensitive assignment
            is at least as concerning as an earlier one, and the chain
            ``p='/etc/hosts'; p='/etc/shadow'`` must surface the shadow.
          * If the new value is benign and the current sensitive, keep
            the sensitive value (Python would last-wins to benign, but
            statically we cannot prove the new value executes and we
            err on the side of blocking the path the attacker reached
            for).
          * If both are benign, latest seen wins."""
        bucket = string_bindings_all.setdefault(name, [])
        if value not in bucket:
            bucket.append(value)
        cur = string_bindings.get(name)
        if cur is None:
            string_bindings[name] = value
            return
        if _looks_sensitive(value):
            string_bindings[name] = value
            return
        if _looks_sensitive(cur):
            return
        string_bindings[name] = value

    def _extract_string_literal(node, _depth = 0):
        """Strict literal-string extraction: no name binding lookup,
        no ``os.path.join`` resolution. Used at sites where conservative
        "dynamic means allow" behaviour is required for non-regression
        (e.g. the trusted-host check, where ``url = some_input;
        requests.get(url)`` must continue to pass through to the host
        gate rather than getting eagerly bound to a literal)."""
        if _depth > 64:
            return None
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return node.value
            if isinstance(node.value, bytes):
                # ``open(b'/etc/shadow')`` — bytes are valid path-like
                # objects to ``open()`` so the literal must reach the
                # sensitive-path gate too. Strict UTF-8 to avoid
                # masking junk.
                try:
                    return node.value.decode("utf-8")
                except UnicodeDecodeError:
                    return None
            if isinstance(node.value, (int, float)):
                return str(node.value)
            return None
        if isinstance(node, ast.NamedExpr):
            # Walrus (``open((p := '/etc/shadow'))``): resolve the RHS.
            return _extract_string_literal(node.value, _depth + 1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            # Flatten left-leaning ``+`` chains iteratively so a long
            # concat ``v0+v1+...+v64+'/etc/shadow'`` does not blow the
            # depth cap (each level adds 1, so the recursive form
            # fails closed at 64 operands).
            operands: list[ast.AST] = []
            cur = node
            while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Add):
                operands.append(cur.right)
                cur = cur.left
            operands.append(cur)
            operands.reverse()
            parts: list[str] = []
            for op in operands:
                s = _extract_string_literal(op, _depth + 1)
                if s is None:
                    return None
                parts.append(s)
            return "".join(parts)
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
                elif isinstance(v, ast.FormattedValue):
                    inner = _extract_string_literal(v.value, _depth + 1)
                    if inner is None:
                        return None
                    parts.append(inner)
                else:
                    return None
            return "".join(parts)
        return None

    def _extract_string_from_node(node, _depth = 0):
        """Extract a plain string value from an AST node when it can be
        resolved statically.

        Handles:
          * ``ast.Constant`` strings (unchanged from prior behaviour).
          * Numeric ``ast.Constant`` values stringified, used inside
            f-strings (``f'/proc/{1}/environ'``).
          * ``ast.BinOp(ast.Add)`` joining two resolvable string operands.
            Closes ``open('/etc/' + 'shadow')`` style dynamic paths.
          * ``ast.JoinedStr`` (f-strings) whose ``FormattedValue`` parts
            are themselves resolvable, including numeric constants.
          * ``ast.Name`` lookups against a name -> literal pre-pass so
            ``p = '/etc/shadow'; open(p)`` resolves.
          * ``os.path.join('/etc', 'shadow')`` and
            ``os.path.expanduser('~/...')`` so common stdlib path
            helpers do not hide a sensitive target.

        Resolution is depth-capped so adversarial deeply-nested
        ``'a' + ('b' + ('c' + ...))`` cannot blow the stack. The cap
        (64) sits well below CPython's default recursion limit and
        comfortably above any realistic credential-path concatenation
        (the longest sensitive path is roughly 30 chars).
        Returns ``None`` whenever any subpart fails to resolve.
        """
        if _depth > 64:
            return None
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return node.value
            if isinstance(node.value, bytes):
                # ``open(b'/etc/shadow')`` -- bytes paths are valid
                # PathLike for ``open()``. Decode strictly so non-UTF-8
                # junk does not mask the gate.
                try:
                    return node.value.decode("utf-8")
                except UnicodeDecodeError:
                    return None
            if isinstance(node.value, (int, float)):
                return str(node.value)
            return None
        if isinstance(node, ast.Name):
            return string_bindings.get(node.id)
        if isinstance(node, ast.NamedExpr):
            # Walrus ``(p := '/etc/shadow')``: resolve and record the
            # binding so later uses of ``p`` also resolve.
            val = _extract_string_from_node(node.value, _depth + 1)
            if val is not None and isinstance(node.target, ast.Name):
                string_bindings.setdefault(node.target.id, val)
            return val
        if isinstance(node, ast.IfExp):
            # Ternary ``'/etc/shadow' if cond else 'data.txt'``: either
            # branch can execute at runtime, so a sensitive value in
            # ANY branch must reach the gate. Prefer the sensitive one
            # so the downstream check fires; fall back to whichever
            # branch resolves.
            body_val = _extract_string_from_node(node.body, _depth + 1)
            orelse_val = _extract_string_from_node(node.orelse, _depth + 1)
            if body_val is not None and _looks_sensitive(body_val):
                return body_val
            if orelse_val is not None and _looks_sensitive(orelse_val):
                return orelse_val
            return body_val if body_val is not None else orelse_val
        if isinstance(node, ast.Subscript):
            # ``['/etc/shadow'][0]`` and ``{'k':'/etc/shadow'}['k']``
            # are statically resolvable index expressions. Attempt the
            # literal value lookup; otherwise return any sensitive
            # candidate in the container so the gate still fires.
            #
            # ``ast.Index`` was folded in Python 3.9 -- on older
            # grammars the slice node would itself be an ``ast.Index``
            # wrapping the constant. Strip the wrapper if present.
            key_node = node.slice
            if isinstance(key_node, getattr(ast, "Index", tuple())):
                key_node = key_node.value
            container = node.value
            if isinstance(container, (ast.List, ast.Tuple)):
                # Indexed list / tuple of literals: prefer the indexed
                # element when the index is a static int; otherwise
                # take any sensitive element so the gate fires.
                if isinstance(key_node, ast.Constant) and isinstance(
                    key_node.value, int
                ):
                    idx = key_node.value
                    if -len(container.elts) <= idx < len(container.elts):
                        v = _extract_string_from_node(container.elts[idx], _depth + 1)
                        if v is not None:
                            return v
                for elt in container.elts:
                    v = _extract_string_from_node(elt, _depth + 1)
                    if v is not None and _looks_sensitive(v):
                        return v
                for elt in container.elts:
                    v = _extract_string_from_node(elt, _depth + 1)
                    if v is not None:
                        return v
                return None
            if isinstance(container, ast.Dict):
                # Indexed dict of literals: prefer the value at the
                # static key; otherwise return any sensitive value.
                if isinstance(key_node, ast.Constant):
                    for k_node, v_node in zip(container.keys, container.values):
                        if (
                            isinstance(k_node, ast.Constant)
                            and k_node.value == key_node.value
                        ):
                            v = _extract_string_from_node(v_node, _depth + 1)
                            if v is not None:
                                return v
                for v_node in container.values:
                    v = _extract_string_from_node(v_node, _depth + 1)
                    if v is not None and _looks_sensitive(v):
                        return v
                return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            # Flatten left-leaning ``+`` chains iteratively to avoid
            # the recursion depth cap rejecting long concat bypasses
            # like ``open(v0+v1+...+v64+'/etc/shadow')``.
            operands: list[ast.AST] = []
            cur = node
            while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Add):
                operands.append(cur.right)
                cur = cur.left
            operands.append(cur)
            operands.reverse()
            parts: list[str] = []
            for op in operands:
                s = _extract_string_from_node(op, _depth + 1)
                if s is None:
                    return None
                parts.append(s)
            return "".join(parts)
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
                elif isinstance(v, ast.FormattedValue):
                    inner = _extract_string_from_node(v.value, _depth + 1)
                    if inner is None:
                        return None
                    parts.append(inner)
                else:
                    return None
            return "".join(parts)
        if isinstance(node, ast.Call):
            # ``os.path.join(a, b, ...)`` and ``os.path.expanduser(s)``
            # are the two stdlib path-building primitives that commonly
            # appear in attacker payloads; resolve them when all inputs
            # are static.
            fq_chain = []
            cur = node.func
            while isinstance(cur, ast.Attribute):
                fq_chain.insert(0, cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                fq_chain.insert(0, cur.id)
            fq = ".".join(fq_chain) if fq_chain else ""
            # Match ``X.join(...)`` where X is any tracked alias of
            # ``os.path`` / ``posixpath`` / ``ntpath`` (handles
            # ``import os as o; o.path.join``, ``from os import path``,
            # ``from os import path as op``, ``import posixpath as pp``).
            is_path_join = (
                fq in ("os.path.join", "posixpath.join", "ntpath.join")
                or (
                    fq.endswith(".join")
                    and fq[: -len(".join")] in os_path_module_aliases
                )
                or (
                    isinstance(node.func, ast.Name)
                    and node.func.id in bare_path_join_aliases
                )
            )
            if is_path_join and node.args:
                parts = []
                for arg in node.args:
                    s = _extract_string_from_node(arg, _depth + 1)
                    if s is None:
                        return None
                    parts.append(s)
                if not parts:
                    return None
                joined = parts[0]
                for p in parts[1:]:
                    if p.startswith(("/", "\\")):
                        joined = p
                    elif joined.endswith(("/", "\\")):
                        joined = joined + p
                    else:
                        joined = joined + "/" + p
                return joined
            is_path_expanduser = (
                fq == "os.path.expanduser"
                or (
                    fq.endswith(".expanduser")
                    and fq[: -len(".expanduser")] in os_path_module_aliases
                )
                or (
                    isinstance(node.func, ast.Name)
                    and node.func.id in bare_path_expanduser_aliases
                )
            )
            if is_path_expanduser and len(node.args) == 1:
                return _extract_string_from_node(node.args[0], _depth + 1)
        return None

    def _run_string_binding_prepass(subtree: ast.AST) -> None:
        """Collect simple ``name = 'literal'`` string assignments and
        ``name = eval`` / ``name = exec`` function aliases from
        ``subtree``. Idempotent and additive: callable on the outer
        module AST and again on each eval / exec literal payload so
        ``exec("p='/etc/shadow'\\nopen(p)")`` is not a free bypass.

        Records every literal so multiple-assignment bypasses (``p =
        '/tmp/safe'; p = '/etc/shadow'; open(p)``) cannot dodge the
        gate by ordering -- the sensitive-shape preference in
        ``_record_string_binding`` picks the dangerous value.

        Also resolves:

        * Tuple / list unpacking destructuring (``(a, b) = ('/etc',
          'shadow')`` and ``p, = ['/etc/shadow']``) element-wise.
        * Pathlib constructor assignments (``p = Path('/etc/shadow');
          p.read_text()``) so the bound name resolves to the path
          string when later referenced by the file-read or shutil gate.
        """
        for _assign in ast.walk(subtree):
            # Walrus (``p := '/etc/shadow'``) is an expression that
            # binds, not an Assign. Handle it here so a walrus inside
            # an eval / exec payload (or any expression context) is
            # surfaced by the pre-pass too.
            if isinstance(_assign, ast.NamedExpr) and isinstance(
                _assign.target, ast.Name
            ):
                _val = _extract_string_from_node(_assign.value)
                if _val is None:
                    _val = _extract_pathlib_target(
                        _assign.value,
                        path_class_aliases_prepass,
                        pathlib_module_aliases_prepass,
                    )
                if _val is not None:
                    _record_string_binding(_assign.target.id, _val)
                continue
            # Annotated assignment (``path: str = '/etc/shadow'``) is
            # an ast.AnnAssign, not an ast.Assign. Same surface: a
            # single Name target bound to a single value.
            if (
                isinstance(_assign, ast.AnnAssign)
                and isinstance(_assign.target, ast.Name)
                and _assign.value is not None
            ):
                _val = _extract_string_from_node(_assign.value)
                if _val is None:
                    _val = _extract_pathlib_target(
                        _assign.value,
                        path_class_aliases_prepass,
                        pathlib_module_aliases_prepass,
                    )
                if _val is not None:
                    _record_string_binding(_assign.target.id, _val)
                continue
            if not isinstance(_assign, ast.Assign):
                continue
            # Chained assignment ``a = b = '/etc/shadow'`` is one Assign
            # node with multiple targets. Resolve the value once and
            # bind every Name target -- ``open(a)`` and ``open(b)``
            # both have to flow through the gate.
            if len(_assign.targets) > 1:
                _val = _extract_string_from_node(_assign.value)
                if _val is None:
                    _val = _extract_pathlib_target(
                        _assign.value,
                        path_class_aliases_prepass,
                        pathlib_module_aliases_prepass,
                    )
                if _val is not None:
                    for _tgt in _assign.targets:
                        if isinstance(_tgt, ast.Name):
                            _record_string_binding(_tgt.id, _val)
                continue
            if len(_assign.targets) == 1:
                _target = _assign.targets[0]
                if isinstance(_target, ast.Name):
                    _val = _extract_string_from_node(_assign.value)
                    if _val is None:
                        # Pathlib fallback: ``p = Path('/etc/shadow')`` /
                        # ``p = pathlib.PosixPath('/proc/self/environ')`` /
                        # ``import pathlib as pl; p = pl.Path('/...')``.
                        # Uses the per-tree alias sets built earlier so
                        # ``import pathlib as pl`` and ``from pathlib
                        # import Path as P`` both resolve.
                        _val = _extract_pathlib_target(
                            _assign.value,
                            path_class_aliases_prepass,
                            pathlib_module_aliases_prepass,
                        )
                    if _val is not None:
                        _record_string_binding(_target.id, _val)
                    elif isinstance(_assign.value, ast.Name) and _assign.value.id in (
                        "eval",
                        "exec",
                    ):
                        eval_exec_aliases.setdefault(_target.id, _assign.value.id)
                elif isinstance(_target, (ast.Tuple, ast.List)) and isinstance(
                    _assign.value, (ast.Tuple, ast.List)
                ):
                    if len(_target.elts) == len(_assign.value.elts):
                        for _tgt_e, _val_e in zip(_target.elts, _assign.value.elts):
                            if isinstance(_tgt_e, ast.Name):
                                _v = _extract_string_from_node(_val_e)
                                if _v is not None:
                                    _record_string_binding(_tgt_e.id, _v)

    # The initial pre-pass call moves to AFTER ``_extract_pathlib_target``
    # is defined so the pathlib fallback resolves (Python closure cell
    # binding rule: ``_run_string_binding_prepass`` looks up the name
    # in the enclosing scope at CALL time, which must be after the
    # ``def`` site runs).

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

    def _join_path_parts(parts):
        """Stitch path parts the way ``pathlib.Path(*parts)`` does for
        statically-resolvable string segments.

        Mirrors pathlib's absolute-segment-reset semantics: when a later
        part starts with ``/`` or a drive letter, it discards everything
        accumulated so far. ``Path('/tmp', '/etc/shadow')`` resolves to
        ``/etc/shadow`` at runtime; this helper does the same."""
        if not parts:
            return None
        out = parts[0]
        for p in parts[1:]:
            if p.startswith(("/", "\\")) or (
                len(p) >= 2 and p[1] == ":" and p[0].isalpha()
            ):
                out = p
                continue
            if out.endswith(("/", "\\")):
                out = out + p.lstrip("/\\")
            else:
                out = out + "/" + p.lstrip("/\\")
        return out

    def _fq_chain_name(func):
        """Return the dotted FQ chain for an attribute / name expression,
        or empty string if the chain stops at something other than a Name."""
        parts: list[str] = []
        cur = func
        while isinstance(cur, ast.Attribute):
            parts.insert(0, cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.insert(0, cur.id)
        return ".".join(parts) if parts else ""

    # Pathlib methods that return ``self`` unchanged for the purposes
    # of static path matching: tilde expansion, symlink resolution, and
    # absolutification do not change which path the read will hit.
    _PATHLIB_PASS_THROUGH = frozenset({"expanduser", "resolve", "absolute"})
    # Pathlib concrete classes that behave like Path for our purposes.
    _PATHLIB_PATH_CLASSES = frozenset(
        {
            "Path",
            "PurePath",
            "PosixPath",
            "WindowsPath",
            "PurePosixPath",
            "PureWindowsPath",
        }
    )

    def _extract_pathlib_target(node, path_aliases, pathlib_aliases, _depth = 0):
        """Statically resolve a pathlib expression to its target path
        string, or None if any subpart is not resolvable.

        Recognises (with depth cap):
          * Plain string literals (delegated to ``_extract_string_from_node``).
          * ``Path('/etc/shadow')`` and aliased ``P('/etc/shadow')`` /
            ``pl.Path('/etc/shadow')`` / ``PosixPath('/etc/shadow')``.
          * Multi-part construction ``Path('/etc', 'shadow')``.
          * ``Path('/etc').joinpath('shadow')`` (one or more parts).
          * ``Path('/etc') / 'shadow'`` (``__truediv__`` chain).
          * ``Path.home()`` resolves to ``~`` so subsequent ``/`` or
            ``.joinpath()`` reach the home-prefix regex.
          * ``.expanduser()`` / ``.resolve()`` / ``.absolute()``
            pass-through.
        """
        if _depth > 32:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return string_bindings.get(node.id)
        if isinstance(node, ast.Call):
            # Pass-through methods on a pathlib object (.expanduser(),
            # .resolve(), .absolute()): return the receiver path.
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _PATHLIB_PASS_THROUGH
            ):
                return _extract_pathlib_target(
                    node.func.value, path_aliases, pathlib_aliases, _depth + 1
                )
            if isinstance(node.func, ast.Attribute) and node.func.attr == "joinpath":
                base = _extract_pathlib_target(
                    node.func.value, path_aliases, pathlib_aliases, _depth + 1
                )
                if base is None:
                    return None
                parts = [base]
                for arg in node.args:
                    s = _extract_pathlib_target(
                        arg, path_aliases, pathlib_aliases, _depth + 1
                    )
                    if s is None:
                        return None
                    parts.append(s)
                return _join_path_parts(parts)
            ctor_fq = _fq_chain_name(node.func)
            # ``Path.home()`` (and aliases) resolves to ``~`` so
            # ``Path.home() / '.aws/credentials'`` reaches the
            # ``~/.aws/credentials`` home-anchored regex below.
            if ctor_fq in {f"{a}.home" for a in path_aliases} or ctor_fq in {
                f"{a}.Path.home" for a in pathlib_aliases
            }:
                return "~"
            is_path_ctor = ctor_fq in path_aliases or any(
                ctor_fq == f"{alias}.{cls}"
                for alias in pathlib_aliases
                for cls in _PATHLIB_PATH_CLASSES
            )
            if is_path_ctor and node.args:
                parts = []
                for arg in node.args:
                    s = _extract_pathlib_target(
                        arg, path_aliases, pathlib_aliases, _depth + 1
                    )
                    if s is None:
                        return None
                    parts.append(s)
                return _join_path_parts(parts)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = _extract_pathlib_target(
                node.left, path_aliases, pathlib_aliases, _depth + 1
            )
            right = _extract_pathlib_target(
                node.right, path_aliases, pathlib_aliases, _depth + 1
            )
            if left is not None and right is not None:
                return _join_path_parts([left, right])
        # Last-ditch: BinOp.Add of string constants, JoinedStr, etc.
        return _extract_string_from_node(node)

    _PATH_RECEIVER_READ_METHODS = frozenset({"open", "read_text", "read_bytes"})

    # ``_extract_pathlib_target`` is now defined; run the string-binding
    # pre-pass so the pathlib fallback inside it resolves.
    _run_string_binding_prepass(tree)

    def _eval_exec_call_name(func, builtins_aliases):
        """Match ``eval`` / ``exec`` invocations including:

          * Bare ``eval`` / ``exec``.
          * Qualified forms ``builtins.exec``, ``__builtins__.eval``,
            and any tracked alias of ``builtins`` (``import builtins as b``).
          * ``from builtins import exec as e`` aliases (tracked per
            visitor in ``shell_exec_aliases``).
          * Simple ``e = eval`` assignment aliases collected by the
            pre-pass into ``eval_exec_aliases``.

        Returns the bare function name (``eval`` or ``exec``) when
        recognised, else None."""
        if isinstance(func, ast.Name):
            if func.id in ("eval", "exec"):
                return func.id
            return eval_exec_aliases.get(func.id)
        if (
            isinstance(func, ast.Attribute)
            and func.attr in ("eval", "exec")
            and isinstance(func.value, ast.Name)
            and func.value.id in builtins_aliases
        ):
            return func.attr
        return None

    def _resolve_dynamic_module_name(node):
        """Return the module string for dynamic import expressions.

        Recognises:
          * ``__import__('os')``
          * ``importlib.import_module('os')``
          * bare ``import_module('os')`` (after ``from importlib import
            import_module``)

        Returns the literal first-argument string when matched, else
        ``None``. Used to ensure ``__import__('os').system(...)`` and
        ``m = importlib.import_module('os'); m.system(...)`` flow
        through the same shell-escape gate as ``import os; os.system(...)``.
        """
        if not isinstance(node, ast.Call) or not node.args:
            return None
        arg0 = node.args[0]
        if not (isinstance(arg0, ast.Constant) and isinstance(arg0.value, str)):
            return None
        f = node.func
        if isinstance(f, ast.Name) and f.id in ("__import__", "import_module"):
            return arg0.value
        if isinstance(f, ast.Attribute) and f.attr == "import_module":
            return arg0.value
        return None

    # Keyword argument names that carry command content (as opposed to
    # control flags like check=True, text=True, capture_output=True).
    _CMD_KWARGS = frozenset({"args", "command", "executable", "path", "file"})

    def _check_args_for_blocked(args_nodes):
        """Check if any call arguments contain blocked commands or
        clear-cut credential / process-state paths.

        Mirrors the bash side's combined ``_find_blocked_commands`` +
        ``_find_sensitive_paths`` so e.g. ``os.system('cat ~/.ssh/id_rsa')``
        is caught by the same gate as ``bash $ cat ~/.ssh/id_rsa``.
        """
        found = set()
        for arg in args_nodes:
            s = _extract_string_from_node(arg)
            if s is not None:
                found |= _find_blocked_commands(s)
                found |= _find_sensitive_paths(s)
            strs = _extract_strings_from_list(arg)
            for s in strs:
                found |= _find_blocked_commands(s)
                found |= _find_sensitive_paths(s)
        return found

    class SignalEscapeVisitor(ast.NodeVisitor):
        def __init__(self):
            self.imports_signal = False
            self.signal_aliases = {"signal"}
            self.os_aliases = {"os"}
            self.subprocess_aliases = {"subprocess"}
            # Maps bare function names to their fully-qualified form
            # for from-import tracking (e.g. "system" -> "os.system")
            self.shell_exec_aliases: dict[str, str] = {}
            # Builtins aliases so ``builtins.exec`` / ``__builtins__.eval``
            # and ``import builtins as b; b.exec(...)`` flow through the
            # same recursion guard as the bare-name forms.
            self.builtins_aliases = {"builtins", "__builtins__"}
            # Names that resolve to ``importlib.import_module`` so
            # ``from importlib import import_module as IM; IM('os')...``
            # flows through ``_resolve_dynamic_module`` the same as
            # ``import importlib; importlib.import_module('os')...``.
            self.import_module_aliases = {"import_module"}
            self.loop_depth = 0
            # Cap recursion into nested eval/exec literals; an adversarial
            # ``eval("eval('eval(...)')")`` should not blow the stack.
            self._eval_depth = 0

        def visit_Import(self, node):
            for alias in node.names:
                if alias.name == "signal":
                    self.imports_signal = True
                    if alias.asname:
                        self.signal_aliases.add(alias.asname)
                elif alias.name == "os":
                    self.os_aliases.add(alias.asname or "os")
                elif alias.name == "builtins":
                    self.builtins_aliases.add(alias.asname or "builtins")
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
                # Track from-imports of dangerous functions
                for alias in node.names:
                    fq = f"{node.module}.{alias.name}"
                    if fq in _SHELL_EXEC_FUNCS:
                        self.shell_exec_aliases[alias.asname or alias.name] = fq
            elif node.module == "builtins":
                # ``from builtins import exec as e`` / ``eval as e``
                # registers the alias for both the literal-payload
                # recursion (via eval_exec_aliases) and the builtins
                # qualified-call resolution.
                for alias in node.names:
                    if alias.name in ("eval", "exec"):
                        eval_exec_aliases[alias.asname or alias.name] = alias.name
            elif node.module == "importlib":
                # ``from importlib import import_module as IM`` so a
                # later ``IM('os').system(...)`` flows through the same
                # dynamic-import gate as ``importlib.import_module('os')``.
                for alias in node.names:
                    if alias.name == "import_module":
                        self.import_module_aliases.add(alias.asname or alias.name)
            self.generic_visit(node)

        def visit_While(self, node):
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_For(self, node):
            self.loop_depth += 1
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_Assign(self, node):
            # Track ``m = __import__('os')`` and
            # ``m = importlib.import_module('os')`` so a subsequent
            # ``m.system(...)`` / ``m.popen(...)`` flows through the
            # os/subprocess alias detection unchanged.
            dyn = self._resolve_dynamic_module(node.value)
            if dyn == "os":
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        self.os_aliases.add(tgt.id)
            elif dyn == "subprocess":
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        self.subprocess_aliases.add(tgt.id)

            # Bare module rebinding (``m = os`` / ``r = subprocess``):
            # propagate the source alias set so a later ``m.system(...)``
            # is caught by the same os/subprocess gate as the direct call.
            if isinstance(node.value, ast.Name):
                src = node.value.id
                if src in self.os_aliases:
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            self.os_aliases.add(tgt.id)
                elif src in self.subprocess_aliases:
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            self.subprocess_aliases.add(tgt.id)

            # Method rebinding (``p = os.popen`` / ``r = subprocess.run``):
            # the bound name now points at a shell-exec function so a
            # later ``p('sudo whoami')`` must flow through the
            # shell-escape gate. Track it under ``shell_exec_aliases``
            # alongside the existing from-import path.
            elif isinstance(node.value, ast.Attribute) and isinstance(
                node.value.value, ast.Name
            ):
                recv = node.value.value.id
                attr = node.value.attr
                fq = None
                if recv in self.os_aliases:
                    fq = f"os.{attr}"
                elif recv in self.subprocess_aliases:
                    fq = f"subprocess.{attr}"
                if fq and fq in _SHELL_EXEC_FUNCS:
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            self.shell_exec_aliases[tgt.id] = fq

            self.generic_visit(node)

        def _resolve_dynamic_module(self, node):
            """Visitor-aware dynamic-import detection: recognises
            everything :func:`_resolve_dynamic_module_name` does plus
            tracked ``from importlib import import_module as IM``
            aliases stored on ``self.import_module_aliases``."""
            mod = _resolve_dynamic_module_name(node)
            if mod is not None:
                return mod
            if isinstance(node, ast.Call) and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    if (
                        isinstance(node.func, ast.Name)
                        and node.func.id in self.import_module_aliases
                    ):
                        return arg0.value
            return None

        def visit_Call(self, node):
            func = node.func

            # --- eval / exec body inspection --------------------------
            # If a payload is a statically-resolvable string we parse it
            # and recurse so the inner code is checked by all the same
            # detectors (signal tampering, shell escape, sensitive files,
            # network policy). If the payload is not statically resolvable
            # we flag it as a dynamic shell-escape candidate — eval/exec
            # of runtime data is the classic injection vector.
            eval_exec_name = _eval_exec_call_name(func, self.builtins_aliases)
            if eval_exec_name is not None:
                if node.args:
                    payload = _extract_string_from_node(node.args[0])
                    if payload is None:
                        # Dynamic payload: classic injection vector.
                        shell_escapes.append(
                            {
                                "type": "shell_escape_dynamic",
                                "line": node.lineno,
                                "description": (
                                    f"{eval_exec_name}() called with non-literal "
                                    "argument (potential code-injection escape)"
                                ),
                            }
                        )
                    elif self._eval_depth >= 3:
                        # Fail-closed at the recursion cap so an attacker
                        # cannot bypass inspection by wrapping the payload
                        # in four-plus nested literal eval/exec layers.
                        shell_escapes.append(
                            {
                                "type": "shell_escape_dynamic",
                                "line": node.lineno,
                                "description": (
                                    f"{eval_exec_name}() literal payload nesting "
                                    "exceeds sandbox inspection depth"
                                ),
                            }
                        )
                    else:
                        try:
                            inner_tree = ast.parse(payload, mode = "exec")
                        except SyntaxError:
                            inner_tree = None
                        if inner_tree is not None:
                            # Re-run the string-binding pre-pass on the
                            # payload so ``exec("p='/etc/shadow'\\nopen(p)")``
                            # surfaces ``p``'s literal before the
                            # ``open(p)`` visit. Without this the inner
                            # ``Name('p')`` lookup misses and the read
                            # is treated as dynamic-and-allowed.
                            _run_alias_prepass(inner_tree)
                            _run_string_binding_prepass(inner_tree)
                            self._eval_depth += 1
                            try:
                                self.visit(inner_tree)
                            finally:
                                self._eval_depth -= 1

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
                        if _ast_name_matches(
                            node.args[0], ("SIGALRM", "signal.SIGALRM")
                        ):
                            signal_tampering.append(
                                {
                                    "type": "signal_handler_override",
                                    "line": node.lineno,
                                    "description": "Overrides SIGALRM handler",
                                }
                            )
                elif func_name in ("signal.setitimer", "setitimer"):
                    if len(node.args) >= 1:
                        if _ast_name_matches(
                            node.args[0], ("ITIMER_REAL", "signal.ITIMER_REAL")
                        ):
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
            # Resolve the fully qualified function name for os.*/subprocess.*
            shell_func = None
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name):
                    if func.value.id in self.os_aliases:
                        shell_func = f"os.{func.attr}"
                    elif func.value.id in self.subprocess_aliases:
                        shell_func = f"subprocess.{func.attr}"
                else:
                    # Inline dynamic import:
                    #   __import__('os').system(...)
                    #   importlib.import_module('os').popen(...)
                    #   IM('os').system(...)  (IM is a from-import alias)
                    # No intermediate name binding so the Name branch
                    # above misses it; resolve the receiver here.
                    dyn = self._resolve_dynamic_module(func.value)
                    if dyn == "os":
                        shell_func = f"os.{func.attr}"
                    elif dyn == "subprocess":
                        shell_func = f"subprocess.{func.attr}"
            elif isinstance(func, ast.Name):
                # Check from-import aliases: from os import system; system(...)
                shell_func = self.shell_exec_aliases.get(func.id)

            if shell_func and shell_func in _SHELL_EXEC_FUNCS:
                # Expand **kwargs dicts to inspect their keys
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

                cmd_kw_values = [
                    v for k, v in expanded_kwargs.items() if k in _CMD_KWARGS
                ]
                all_call_args = list(node.args) + cmd_kw_values
                blocked_in_args = _check_args_for_blocked(all_call_args)

                if has_opaque_kwargs:
                    # Can't inspect dynamic **kwargs -- flag as unsafe
                    shell_escapes.append(
                        {
                            "type": "shell_escape_dynamic",
                            "line": node.lineno,
                            "description": (
                                f"{shell_func}() called with dynamic **kwargs"
                            ),
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
                    # Only flag dynamic args for functions that interpret
                    # strings as shell commands, or when shell= might be
                    # enabled.  Treat any non-literal-False shell= value
                    # as potentially True (conservative).
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
                        isinstance(shell_node, ast.Constant)
                        and shell_node.value is False
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
                                return all(
                                    _extract_string_from_node(e) is not None
                                    for e in n.elts
                                )
                            return False

                        has_non_literal = any(
                            not _is_safe_literal(a) for a in all_call_args
                        )
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
                # Only flag BaseException and TimeoutError, NOT Exception.
                # except Exception does not catch SystemExit or
                # KeyboardInterrupt, so it cannot suppress timeout
                # enforcement.  Flagging Exception causes false positives
                # on normal error-handling patterns.
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

    # Static host policy: block metadata hosts and any literal host outside
    # the trusted allowlist; uploads blocked regardless of host. Dynamic hosts
    # are caught by the bash blocklist instead.
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
    _SENSITIVE_FILE_RE = re.compile(
        r"^/proc/(?:self|\d+)/(?:environ|cmdline|task/\d+/environ)$"
    )

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
                if (
                    isinstance(v, ast.Call)
                    and isinstance(v.func, ast.Name)
                    and v.func.id == "open"
                ):
                    return True
                if isinstance(v, ast.Constant) and isinstance(
                    v.value, (bytes, bytearray)
                ):
                    return True
        return False

    # Bare method-name fallback (`x.upload_file(...)`) is intentionally fuzzy,
    # but should only fire when huggingface_hub / hf_api is actually imported
    # somewhere in the snippet -- otherwise paramiko.upload_file, boto3
    # create_commit, etc. hit a false positive. We pre-scan for the imports.
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
                # __import__('huggingface_hub'), importlib.import_module('huggingface_hub'),
                # and bare import_module('huggingface_hub') (via `from importlib import ...`).
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
        """Return the HF upload method name (`upload_file`, ...) or None.

        Catches `HfApi().upload_file(...)` (Attribute) and
        `from huggingface_hub import upload_file; upload_file(...)` (Name).
        The bare-name branch fires only when an HF import is in scope, mirroring
        the Attribute branch's gating so paramiko/boto3 do not false-positive.
        """
        if not _hf_in_scope:
            return None
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in _UPLOAD_HF_METHODS:
            return f.attr
        if isinstance(f, ast.Name) and f.id in _UPLOAD_HF_METHODS:
            return f.id
        return None

    # Kwargs that ship a credential over the wire. Sandbox env strips HF_TOKEN
    # / WANDB_API_KEY / AWS_* up front, so any value here is hard-coded or
    # lifted from the parent process.
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
        """True if any node in the subtree resolves to an env / process read.

        Walking the subtree (not just the root) means wrapper calls like
        `str(os.environ)`, `json.dumps(os.environ)`, or
        `'-'.join(os.environ.values())` are caught too.

        Covers: `os.environ`, `os.environ[K]`, `os.environ.get(K)`, `os.getenv(K)`,
        bare `getenv(K)` (after `from os import getenv`), and
        `subprocess.{run,check_output,Popen,getoutput,getstatusoutput}` which
        the LLM could use to lift parent env via `printenv` / `env` / `set`.
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
        if isinstance(node, ast.Constant) and isinstance(
            node.value, (bytes, bytearray)
        ):
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
        readers, and (c) the path argument is a sandbox-local literal -- a
        relative string with no `..`, an `open(<literal>)`, or inline bytes.
        Dynamic / variable paths are rejected; the policy cannot prove safety
        statically and the cost of a wrong-allow is a credential exfiltration.
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
        def __init__(self):
            super().__init__()
            self._eval_depth = 0
            # Builtins / pathlib alias tracking so the receiver-side
            # pathlib detection and the eval/exec recursion both reach
            # qualified and aliased forms (``builtins.exec``, ``P('/etc/x')``,
            # ``PosixPath(...)``).
            self.builtins_aliases = {"builtins", "__builtins__"}
            self.path_aliases = set(_PATHLIB_PATH_CLASSES)
            self.pathlib_aliases = {"pathlib"}
            # ``from io import FileIO as X`` and ``from codecs import open
            # as X``: a later bare ``X('/etc/shadow')`` flows through the
            # same file-read gate as the qualified call.
            self.file_reader_aliases: set[str] = set()

        def visit_Import(self, node):
            for alias in node.names:
                if alias.name == "pathlib":
                    self.pathlib_aliases.add(alias.asname or "pathlib")
                elif alias.name == "builtins":
                    self.builtins_aliases.add(alias.asname or "builtins")
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if node.module == "pathlib":
                for alias in node.names:
                    if alias.name in _PATHLIB_PATH_CLASSES:
                        self.path_aliases.add(alias.asname or alias.name)
            elif node.module == "builtins":
                for alias in node.names:
                    if alias.name in ("eval", "exec"):
                        eval_exec_aliases[alias.asname or alias.name] = alias.name
            elif node.module in ("io", "codecs"):
                # ``from io import FileIO`` / ``from codecs import open``
                # bind a bare name that is otherwise indistinguishable
                # from any other ``FileIO(...)`` / ``open(...)`` call.
                # The reader's gate uses this set to recognise the
                # alias as a file-read.
                for alias in node.names:
                    if (node.module == "io" and alias.name in ("FileIO", "open")) or (
                        node.module == "codecs" and alias.name == "open"
                    ):
                        self.file_reader_aliases.add(alias.asname or alias.name)
            self.generic_visit(node)

        def visit_Assign(self, node):
            # Module rebinding: ``import pathlib; pl = pathlib``,
            # ``import shutil; sh = shutil`` (and the equivalent for
            # ``io`` / ``codecs``). Mirrors ``SignalEscapeVisitor.visit_Assign``
            # so the file-read / shutil-copy / pathlib gates see the
            # bound alias the same way they see the import-time alias.
            if isinstance(node.value, ast.Name):
                src = node.value.id
                for tgt in node.targets:
                    if not isinstance(tgt, ast.Name):
                        continue
                    if src in self.pathlib_aliases:
                        self.pathlib_aliases.add(tgt.id)
                    if src in shutil_module_aliases:
                        shutil_module_aliases.add(tgt.id)
                    if src in self.builtins_aliases:
                        self.builtins_aliases.add(tgt.id)
                    if src in self.path_aliases:
                        self.path_aliases.add(tgt.id)
            # Method rebinding inside the file-read surface:
            # ``r = pl.Path`` so a later ``r('/etc/shadow').read_text()``
            # flows through the pathlib resolver. The receiver alias
            # for ``shutil.copy`` etc. is handled by the shutil-fq
            # canonicalisation in the gate itself.
            if isinstance(node.value, ast.Attribute) and isinstance(
                node.value.value, ast.Name
            ):
                recv = node.value.value.id
                attr = node.value.attr
                if recv in self.pathlib_aliases and attr in _PATHLIB_PATH_CLASSES:
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            self.path_aliases.add(tgt.id)
            self.generic_visit(node)

        def visit_Call(self, node):
            func = node.func
            # eval/exec payload recursion — see SignalEscapeVisitor for
            # the dual gate. Catches ``exec("open('/etc/shadow').read()")``
            # by parsing the literal payload and walking it through the
            # same sensitive-file / network / upload checks.
            eval_exec_name = _eval_exec_call_name(func, self.builtins_aliases)
            if eval_exec_name is not None:
                if node.args:
                    payload = _extract_string_from_node(node.args[0])
                    if payload is not None:
                        if self._eval_depth >= 3:
                            # Fail-closed at the depth cap so nested literal
                            # ``exec(exec(exec(exec("open('/etc/shadow')"))))``
                            # cannot tunnel past inspection.
                            sensitive_file_reads.append(
                                {
                                    "type": "sensitive_file_read",
                                    "line": getattr(node, "lineno", -1),
                                    "description": (
                                        f"{eval_exec_name}() literal payload nesting "
                                        "exceeds sandbox inspection depth"
                                    ),
                                }
                            )
                        else:
                            try:
                                inner_tree = ast.parse(payload, mode = "exec")
                            except SyntaxError:
                                inner_tree = None
                            if inner_tree is not None:
                                # Mirror SignalEscapeVisitor: re-run
                                # the string-binding pre-pass on the
                                # payload so inner variable assignments
                                # are visible to this visitor too. The
                                # gate currently works because the
                                # other visitor runs first and shares
                                # ``string_bindings``, but making this
                                # site independently correct prevents
                                # a silent regression if visitor order
                                # ever changes.
                                _run_alias_prepass(inner_tree)
                                _run_string_binding_prepass(inner_tree)
                                self._eval_depth += 1
                                try:
                                    self.visit(inner_tree)
                                finally:
                                    self._eval_depth -= 1

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

            # Direct sock.connect((host, port)) bypasses the FQ-prefix branch below.
            # ``sendto`` / ``sendmsg`` / ``connect_ex`` carry the dest
            # ``(host, port)`` tuple the same way ``connect`` does
            # (datagram sockets never call ``.connect()``). Match them
            # all so ``s.sendto(b'x', ('169.254.169.254', 80))`` is
            # gated by the same metadata-host check.
            _SOCKET_DEST_METHODS = {"connect", "connect_ex", "sendto", "sendmsg"}
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _SOCKET_DEST_METHODS
            ):
                # Resolve the host through the strict literal extractor:
                # variable assignments stay opaque to this gate so
                # ``host = some_input; sock.connect((host, 80))`` keeps
                # legitimate dynamic-host tool calls passing through.
                #
                # ``sendto(data, address)`` and ``sendmsg(buffers,
                # ancdata, flags, address)`` carry the address tuple at
                # a non-zero positional index, so scan every positional
                # arg for a ``(host, port)`` tuple shape -- the first
                # match wins.
                host_lit = None
                for a in node.args:
                    if isinstance(a, ast.Tuple) and a.elts:
                        host_lit = _extract_string_literal(a.elts[0])
                        if host_lit:
                            break
                if host_lit is None and node.args:
                    host_lit = _extract_string_literal(node.args[0])
                # Keyword forms: sock.connect(address=(host, port)).
                if host_lit is None:
                    for kw in node.keywords or []:
                        if kw.arg in ("address", "host", "hostname"):
                            v = kw.value
                            if isinstance(v, ast.Tuple) and v.elts:
                                host_lit = _extract_string_literal(v.elts[0])
                            else:
                                host_lit = _extract_string_literal(v)
                            if host_lit:
                                break
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
                            "description": (
                                "Blocked: file upload disallowed in sandbox"
                            ),
                        }
                    )

                # 2) Extract literal host. Three call shapes are handled:
                #
                #   * Host-first APIs whose positional arg 0 is the host
                #     directly (``socket.getaddrinfo('169.254.169.254', 80)``,
                #     ``http.client.HTTPConnection('169.254.169.254')``).
                #   * URL-second APIs whose positional arg 1 is the URL
                #     (``requests.request('GET', 'http://...')``).
                #   * Everything else: positional arg 0 is a URL or
                #     ``(host, port)`` tuple, with keyword fallbacks for
                #     ``url=``, ``address=``, ``host=`` / ``hostname=``.
                _HOST_FIRST_FQ = (
                    "socket.create_connection",
                    "socket.getaddrinfo",
                    "http.client.HTTPConnection",
                    "http.client.HTTPSConnection",
                )
                _URL_SECOND_FQ = ("requests.request", "httpx.request")

                host_arg = None
                url_arg = None

                if node.args:
                    if fq in _URL_SECOND_FQ:
                        # ``requests.request('GET', url='http://...')`` —
                        # positional arg 0 is the HTTP method, not the
                        # URL. Only treat args[1] as the URL; otherwise
                        # leave url_arg/host_arg None so the kw fallback
                        # below picks up ``url=``.
                        if len(node.args) >= 2:
                            url_arg = _extract_string_literal(node.args[1])
                    else:
                        a0 = node.args[0]
                        if isinstance(a0, ast.Tuple) and a0.elts:
                            host_arg = _extract_string_literal(a0.elts[0])
                        elif fq in _HOST_FIRST_FQ:
                            host_arg = _extract_string_literal(a0)
                        else:
                            url_arg = _extract_string_literal(a0)

                # Keyword fallback. ``url=`` and ``address=`` carry the
                # full URL or (host, port); ``host=`` / ``hostname=``
                # carry just the host. Strict literal extraction keeps
                # ``url = some_input; requests.get(url=url)`` flowing
                # through to runtime allow/deny without the static gate
                # eagerly binding the name.
                for kw in node.keywords or []:
                    if kw.arg in ("url", "address"):
                        v = kw.value
                        if isinstance(v, ast.Tuple) and v.elts:
                            if host_arg is None:
                                host_arg = _extract_string_literal(v.elts[0])
                        else:
                            if url_arg is None and host_arg is None:
                                url_arg = _extract_string_literal(v)
                    elif kw.arg in ("host", "hostname"):
                        if host_arg is None:
                            host_arg = _extract_string_literal(kw.value)

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

            # File-read surface detection. Three families are recognised:
            #
            #   * Bare ``open(arg)`` / ``open(file=...)`` and ``io.open``.
            #   * Receiver-side pathlib reads: ``Path(...).open()``,
            #     ``Path(...).open('r')`` (where ``args[0]`` is the MODE,
            #     not the path), ``Path(...).read_text()``, and
            #     ``Path(...).read_bytes()``. The path is extracted from
            #     the receiver expression by ``_extract_pathlib_target``,
            #     which handles ``Path(a, b)``, ``Path().joinpath()``,
            #     ``Path() / arg``, and aliased Path constructors.
            #
            # ``fq`` only resolves when the attribute chain ends in a
            # Name, so ``Path(...).open()`` (with a Call in the chain)
            # short-circuits to ``"open"`` — we accept any Attribute
            # call whose attr is in the path-reader set and pull the
            # actual target from the receiver.
            receiver_read_method = None
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _PATH_RECEIVER_READ_METHODS
            ):
                receiver_read_method = node.func.attr

            # ``io.FileIO`` and ``codecs.open`` are the two stdlib
            # file-reader call shapes that don't end in ``.open`` /
            # ``open()`` but still read an arbitrary path. Treat them
            # as the same gate so ``io.FileIO('/etc/shadow').read()`` is
            # blocked alongside ``open('/etc/shadow')``.
            _EXPLICIT_FILE_READERS = ("io.FileIO", "codecs.open")
            # Third-party file-reader method names that any reasonable
            # ``pandas``/``numpy`` alias exposes (``pd.read_csv`` /
            # ``pandas.read_csv`` / ``np.fromfile`` / ``numpy.loadtxt``).
            # Matched by suffix so the receiver alias does not need to
            # be tracked separately.
            _DATAFRAME_READERS = (
                ".read_csv",
                ".read_table",
                ".read_excel",
                ".read_json",
                ".read_parquet",
                ".read_pickle",
                ".read_feather",
                ".read_orc",
                ".read_hdf",
                ".read_sas",
                ".read_stata",
                ".read_xml",
                ".read_fwf",
                ".read_sql",
                ".fromfile",
                ".loadtxt",
                ".genfromtxt",
            )
            looks_like_dataframe_reader = isinstance(node.func, ast.Attribute) and any(
                fq.endswith(s) for s in _DATAFRAME_READERS
            )
            is_open_call = (
                (
                    isinstance(node.func, ast.Name)
                    and (
                        node.func.id == "open"
                        or node.func.id in self.file_reader_aliases
                    )
                )
                or fq in ("io.open", "pathlib.Path.open")
                or fq in _EXPLICIT_FILE_READERS
                or fq.endswith(".open")
                or looks_like_dataframe_reader
                or receiver_read_method is not None
            )
            if is_open_call:
                path_lit = None

                if receiver_read_method is not None:
                    # For ``Path('/etc/shadow').open('r')`` the positional
                    # arg is the open mode, not the path. Pull the path
                    # exclusively from the receiver to avoid misreading
                    # ``'r'`` as a target.
                    path_lit = _extract_pathlib_target(
                        node.func.value,
                        self.path_aliases,
                        self.pathlib_aliases,
                    )

                if path_lit is None and node.args:
                    # Built-in ``open()`` accepts ``PathLike`` objects, so
                    # ``open(Path('/etc/shadow'))`` and
                    # ``open(Path('/etc') / 'shadow')`` need the pathlib
                    # resolver too — not just plain string literals.
                    path_lit = _extract_pathlib_target(
                        node.args[0], self.path_aliases, self.pathlib_aliases
                    )
                    if path_lit is None:
                        path_lit = _extract_string_from_node(node.args[0])

                # Keyword form. Covers:
                #   * ``open(file=...)`` / ``io.open(file=...)``
                #   * ``pd.read_csv(filepath_or_buffer=...)`` /
                #     ``pd.read_parquet(path=...)`` etc.
                #   * ``np.fromfile(file=...)`` / ``np.loadtxt(fname=...)`` /
                #     ``np.load(file=...)``
                # The keyword set is intentionally broad because the
                # downstream sensitive-path check is the actual gate;
                # extra kwargs just give us additional ways to spot
                # the path argument.
                _FILE_PATH_KWARGS = (
                    "file",
                    "path",
                    "filepath",
                    "filepath_or_buffer",
                    "path_or_buf",
                    "fname",
                    "filename",
                    "io",
                    "buf",
                    "source",
                    "src",
                )
                if path_lit is None:
                    for kw in node.keywords or []:
                        if kw.arg in _FILE_PATH_KWARGS:
                            path_lit = _extract_pathlib_target(
                                kw.value,
                                self.path_aliases,
                                self.pathlib_aliases,
                            )
                            if path_lit is None:
                                path_lit = _extract_string_from_node(kw.value)
                            if path_lit is not None:
                                break

                if path_lit:
                    # Cross-product the projections: backslash-normalised
                    # and path-separator-collapsed (``/etc//shadow``,
                    # ``/etc/./shadow``) so equivalent spellings match.
                    candidates = {path_lit}
                    if "\\" in path_lit:
                        candidates.add(path_lit.replace("\\", "/"))
                    candidates.add(_normalize_path_separators(path_lit))

                    flagged = False
                    for cand in candidates:
                        if any(cand.startswith(p) for p in _SENSITIVE_FILE_PREFIXES):
                            flagged = True
                            break
                        if _SENSITIVE_FILE_RE.match(cand):
                            flagged = True
                            break
                        # The credential / process-state allow-list lives
                        # in ``_find_sensitive_paths`` (Patch B). Reuse it
                        # so ``open('/home/u/.aws/credentials')`` is
                        # blocked the same as the bash equivalent.
                        if _find_sensitive_paths(cand):
                            flagged = True
                            break
                    if flagged:
                        method_label = receiver_read_method or "open"
                        sensitive_file_reads.append(
                            {
                                "type": "sensitive_file_read",
                                "line": getattr(node, "lineno", -1),
                                "description": (
                                    f"{method_label}({path_lit!r}) targets a host "
                                    "identity / credential file; sandboxed code "
                                    "may not read it"
                                ),
                            }
                        )

            # File-copy / file-move APIs read the source path just like
            # ``open()`` does, and the copy gives the attacker a second
            # exfil channel (rename/print/upload the destination). Gate
            # the source argument with the same sensitive-path checks.
            #
            # Matches all three call shapes:
            #   shutil.copy(...) / shutil.copytree(...) etc.
            #   <alias>.copy(...) when ``import shutil as <alias>``
            #   bare copy(...) when ``from shutil import copy [as ...]``
            _FILE_COPY_FUNCS = frozenset(
                {
                    "shutil.copyfile",
                    "shutil.copy",
                    "shutil.copy2",
                    "shutil.copytree",
                    "shutil.move",
                }
            )
            file_copy_fq = None
            if fq in _FILE_COPY_FUNCS:
                file_copy_fq = fq
            elif fq.endswith(_SHUTIL_COPY_NAMES) and isinstance(
                node.func, ast.Attribute
            ):
                # ``sh.copy(...)`` -- check the receiver is a tracked
                # shutil alias. The suffix-match guards against random
                # ``something.copy(...)`` calls on unrelated objects.
                _attr = node.func.attr
                _recv_chain = (
                    fq[: -(len(_attr) + 1)] if _attr in _SHUTIL_COPY_NAMES else ""
                )
                if _recv_chain in shutil_module_aliases and _attr in _SHUTIL_COPY_NAMES:
                    file_copy_fq = f"shutil.{_attr}"
            elif (
                isinstance(node.func, ast.Name)
                and node.func.id in bare_shutil_copy_aliases
            ):
                file_copy_fq = bare_shutil_copy_aliases[node.func.id]
            if file_copy_fq is not None:
                # Use the canonical ``shutil.X`` name in the error
                # description so aliased and from-import bypasses surface
                # with the same identity as the literal form.
                fq = file_copy_fq
                src_lit = None
                if node.args:
                    src_lit = _extract_pathlib_target(
                        node.args[0], self.path_aliases, self.pathlib_aliases
                    )
                    if src_lit is None:
                        src_lit = _extract_string_from_node(node.args[0])
                if src_lit is None:
                    for kw in node.keywords or []:
                        if kw.arg in ("src", "source"):
                            src_lit = _extract_pathlib_target(
                                kw.value,
                                self.path_aliases,
                                self.pathlib_aliases,
                            )
                            if src_lit is None:
                                src_lit = _extract_string_from_node(kw.value)
                            if src_lit is not None:
                                break
                if src_lit:
                    candidates = {src_lit}
                    if "\\" in src_lit:
                        candidates.add(src_lit.replace("\\", "/"))
                    candidates.add(_normalize_path_separators(src_lit))
                    flagged = False
                    for cand in candidates:
                        if any(cand.startswith(p) for p in _SENSITIVE_FILE_PREFIXES):
                            flagged = True
                            break
                        if _SENSITIVE_FILE_RE.match(cand):
                            flagged = True
                            break
                        if _find_sensitive_paths(cand):
                            flagged = True
                            break
                        # Whole-directory exfil: shutil.copytree('~/.ssh',
                        # dst) drags every key out in one call. Reusing
                        # `_find_sensitive_paths` would miss it because
                        # `~/.ssh` (no filename) isn't in the per-file
                        # list. The dir matcher is shutil-specific so
                        # `ls ~/.ssh` (legit) stays allowed.
                        if _matches_sensitive_dir(cand):
                            flagged = True
                            break
                    if flagged:
                        sensitive_file_reads.append(
                            {
                                "type": "sensitive_file_read",
                                "line": getattr(node, "lineno", -1),
                                "description": (
                                    f"{fq}({src_lit!r}, ...) reads a host "
                                    "identity / credential file; sandboxed "
                                    "code may not copy it"
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
        # SyntaxError from ast.parse -- let these through so the subprocess
        # produces a normal Python traceback instead of a misleading
        # "unsafe code detected" message.
        if info.get("error"):
            return None

        reasons = [
            item.get("description", "") for item in info.get("signal_tampering", [])
        ]
        shell_reasons = [
            item.get("description", "") for item in info.get("shell_escapes", [])
        ]
        exception_reasons = [
            item.get("description", "") for item in info.get("exception_catching", [])
        ]
        network_reasons = [
            item.get("description", "") for item in info.get("network_calls", [])
        ]
        file_reasons = [
            item.get("description", "") for item in info.get("sensitive_file_reads", [])
        ]
        all_reasons = [
            r
            for r in reasons
            + shell_reasons
            + exception_reasons
            + network_reasons
            + file_reasons
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


def _cancel_watcher(proc, cancel_event, poll_interval = 0.2):
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


def _python_exec(
    code: str,
    cancel_event = None,
    timeout: int = _EXEC_TIMEOUT,
    session_id: str | None = None,
) -> str:
    """Execute Python code in a subprocess sandbox."""
    if not code or not code.strip():
        return "No code provided."

    # Validate imports and code safety
    error = _check_code_safety(code)
    if error:
        return error

    tmp_path = None
    workdir = _get_workdir(session_id)
    # Snapshot image mtimes so we detect both new and overwritten files.
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
        fd, tmp_path = tempfile.mkstemp(
            suffix = ".py", prefix = "studio_exec_", dir = workdir
        )
        with os.fdopen(fd, "w") as f:
            f.write(code)

        safe_env = _build_safe_env(workdir)
        popen_kwargs = dict(
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            cwd = workdir,
            env = safe_env,
        )
        if sys.platform != "win32":
            popen_kwargs["preexec_fn"] = _sandbox_preexec
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

        # Detect new or overwritten image files and append sentinel for frontend
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
) -> str:
    """Execute a bash command in a subprocess sandbox."""
    if not command or not command.strip():
        return "No command provided."

    # Block dangerous commands (shlex + regex based)
    blocked = _find_blocked_commands(command)
    if blocked:
        return f"Blocked command(s) for safety: {', '.join(sorted(blocked))}"

    # Block direct references to clear-cut credential / process-state
    # paths. Allow-list excludes ~/.gitconfig, ~/.bashrc, ~/.ssh/config,
    # /etc/hosts, ~/.npm/, project-local rc files, etc. so legitimate
    # tool calls (`cat ~/.gitconfig`, `find src/`, `grep -r foo src/`)
    # still work.
    sensitive = _find_sensitive_paths(command)
    if sensitive:
        return (
            f"Blocked: command references credential / process-state paths "
            f"({', '.join(sorted(sensitive))})"
        )

    try:
        workdir = _get_workdir(session_id)
        safe_env = _build_safe_env(workdir)
        popen_kwargs = dict(
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            cwd = workdir,
            env = safe_env,
        )
        if sys.platform != "win32":
            popen_kwargs["preexec_fn"] = _sandbox_preexec
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
