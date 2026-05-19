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

import random
import re
import shlex
import ssl
import subprocess
import sys
import tempfile
import threading
import urllib.request

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
    + r"|/proc/(?:self|\d+)/"
    + r"|/var/spool/"
    + r")"
    + r"[^\s'\";&|`$]*"
    + r"(?:\$\([^)]*\)|`[^`]+`)",
    re.IGNORECASE,
)

_BRACE_EXPANSION_RE = re.compile(r"\{([^{}]*,[^{}]*)\}")


def _normalize_path_separators(text: str) -> str:
    """Collapse ``//`` to ``/``, remove ``/./`` segments, and resolve
    ``/..`` parent-directory traversal so that filesystem-equivalent
    spellings of a sensitive path (``/etc//shadow``, ``/etc/./shadow``,
    ``/etc/apt/../shadow``) match the canonical pattern."""
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
        # absolute or starts with a known root. Reassemble a tilde or
        # ${HOME} prefix afterwards so ``~/.ssh/../.aws/credentials``
        # resolves to ``~/.aws/credentials`` rather than getting eaten.
        for prefix in ("~/", "$HOME/", "${HOME}/", "%USERPROFILE%/"):
            if collapsed.startswith(prefix):
                tail = collapsed[len(prefix):]
                tail = posixpath.normpath("/" + tail).lstrip("/")
                return prefix + tail
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


def _expand_brace_projections(text: str, limit: int = 64) -> set[str]:
    """Return the set of strings reachable from *text* by applying bash
    brace expansion ``{a,b}`` and bounded ``[abc]`` glob character
    classes. Bounded to ``limit`` to keep adversarial inputs from
    fanning out unboundedly."""
    out = {text}
    if "{" not in text and "[" not in text:
        return out
    queue = [text]
    glob_re = re.compile(r"\[([^\]/\\!^]{1,8})\]")
    while queue and len(out) < limit:
        cur = queue.pop()
        brace = _BRACE_EXPANSION_RE.search(cur)
        if brace:
            for alt in brace.group(1).split(","):
                nxt = cur[: brace.start()] + alt + cur[brace.end():]
                if nxt not in out:
                    out.add(nxt)
                    queue.append(nxt)
                    if len(out) >= limit:
                        break
            continue
        klass = glob_re.search(cur)
        if klass:
            for ch in klass.group(1):
                if ch == "-":
                    continue
                nxt = cur[: klass.start()] + ch + cur[klass.end():]
                if nxt not in out:
                    out.add(nxt)
                    queue.append(nxt)
                    if len(out) >= limit:
                        break
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


def _get_workdir(session_id: str | None = None) -> str:
    """Return a per-session sandbox dir at mode 0o700."""
    global _workdirs
    key = session_id or "_default"
    if key not in _workdirs or not os.path.isdir(_workdirs[key]):
        home = os.path.expanduser("~")
        sandbox_root = os.path.join(home, "studio_sandbox")
        if session_id and _SESSION_ID_RE.match(session_id):
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

ALL_TOOLS = [WEB_SEARCH_TOOL, PYTHON_TOOL, TERMINAL_TOOL]


_TIMEOUT_UNSET = object()


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
        if (
            ip.is_private
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
    string_bindings: dict[str, str] = {}
    eval_exec_aliases: dict[str, str] = {}

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
            if isinstance(node.value, (int, float)):
                return str(node.value)
            return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = _extract_string_literal(node.left, _depth + 1)
            right = _extract_string_literal(node.right, _depth + 1)
            if left is not None and right is not None:
                return left + right
            return None
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
            if isinstance(node.value, (int, float)):
                return str(node.value)
            return None
        if isinstance(node, ast.Name):
            return string_bindings.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = _extract_string_from_node(node.left, _depth + 1)
            right = _extract_string_from_node(node.right, _depth + 1)
            if left is not None and right is not None:
                return left + right
            return None
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
            if fq in ("os.path.join", "posixpath.join", "ntpath.join") and node.args:
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
            if fq == "os.path.expanduser" and len(node.args) == 1:
                return _extract_string_from_node(node.args[0], _depth + 1)
        return None

    # Pre-pass: collect simple ``name = 'literal'`` string assignments
    # and ``name = eval`` / ``name = exec`` function aliases so the
    # visitors and ``_extract_string_from_node`` can resolve later uses.
    # Walks the AST in one pass; first assignment wins (mirrors actual
    # execution order well enough for the static gate).
    for _assign in ast.walk(tree):
        if isinstance(_assign, ast.Assign) and len(_assign.targets) == 1:
            _target = _assign.targets[0]
            if isinstance(_target, ast.Name) and _target.id not in string_bindings:
                _val = _extract_string_from_node(_assign.value)
                if _val is not None:
                    string_bindings[_target.id] = _val
                elif (
                    isinstance(_assign.value, ast.Name)
                    and _assign.value.id in ("eval", "exec")
                ):
                    eval_exec_aliases[_target.id] = _assign.value.id

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
        {"Path", "PurePath", "PosixPath", "WindowsPath",
         "PurePosixPath", "PureWindowsPath"}
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
            if (
                ctor_fq in {f"{a}.home" for a in path_aliases}
                or ctor_fq in {f"{a}.Path.home" for a in pathlib_aliases}
            ):
                return "~"
            is_path_ctor = (
                ctor_fq in path_aliases
                or any(
                    ctor_fq == f"{alias}.{cls}"
                    for alias in pathlib_aliases
                    for cls in _PATHLIB_PATH_CLASSES
                )
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
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "connect"
            ):
                # Resolve the host through the strict literal extractor:
                # variable assignments stay opaque to this gate so
                # ``host = some_input; sock.connect((host, 80))`` keeps
                # legitimate dynamic-host tool calls passing through.
                host_lit = None
                if node.args:
                    a0 = node.args[0]
                    if isinstance(a0, ast.Tuple) and a0.elts:
                        host_lit = _extract_string_literal(a0.elts[0])
                    else:
                        host_lit = _extract_string_literal(a0)
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

            is_open_call = (
                (isinstance(node.func, ast.Name) and node.func.id == "open")
                or fq in ("io.open", "pathlib.Path.open")
                or fq.endswith(".open")
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

                # ``open(file=...)`` / ``io.open(file=...)`` keyword form.
                if path_lit is None:
                    for kw in node.keywords or []:
                        if kw.arg in ("file", "path"):
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
            _FILE_COPY_FUNCS = frozenset({
                "shutil.copyfile", "shutil.copy", "shutil.copy2",
                "shutil.copytree", "shutil.move",
            })
            if fq in _FILE_COPY_FUNCS:
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
