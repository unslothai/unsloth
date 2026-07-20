# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Portable, best-effort string-based filesystem screen for the code sandbox.

One pass of simple string / token matching (no AST parse, no kernel calls) that
runs on EVERY platform before the sandbox subprocess is spawned. It flags an
absolute, home (``~``), env-var (``$VAR``) or parent-traversal (``..``) path that
resolves outside the session working directory: for a shell command, the argument
and redirection-target tokens (the command word itself is exempt, since it is an
executable resolved through the sandbox PATH); for python, the quoted string
literals.

It is defense in depth, NOT a real sandbox. A path built at runtime, hidden behind
an escape-encoded literal, or reached through a pre-existing symlink is invisible
to a string scan, so an operand it cannot resolve is treated as allowed and false
positives stay low. Path semantics come from an injected ``posixpath`` / ``ntpath``
module, so the same logic runs on any host and is unit-testable for either platform
on one CI. Rejection messages quote the original operand.

Known, accepted best-effort gaps (documented rather than closed, to keep this a
simple string screen): escape-encoded python literals, a workspace symlink that
points outside (time-of-check/time-of-use), and a nested interpreter payload
(``bash -c '...'``). On Linux the OS-level boundary should be a container / user
separation; this screen is an early, portable rejection layer only.
"""

from __future__ import annotations

import ntpath
import os
import posixpath
import re
import shlex
import tempfile

# Model-convention prefixes the sandbox sitecustomize shim remaps onto the working
# directory at runtime -- but only when they do NOT already exist on the host. We
# mirror that: trusted for python only, and only while absent (see _allowed_roots).
_REMAP_PREFIXES = ("/mnt/data", "/mnt/outputs", "/home/sandbox", "/workspace", "/tmp/outputs")

# Stream devices that are always safe argument / redirection targets.
_DEV_ALLOWED = frozenset(
    {"/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom",
     "/dev/tty", "/dev/stdin", "/dev/stdout", "/dev/stderr"}
)

# Only an explicit OFF disables the screen; unset / auto / on keep it on.
_DISABLE_ENV = "UNSLOTH_STUDIO_SANDBOX_FS_CONFINE"
_OFF_VALUES = frozenset({"0", "false", "off", "no", "disable", "disabled"})

# Shell control tokens produced by the punctuation-aware lexer.
_SEPARATORS = frozenset({";", "|", "&&", "||", "&", "(", ")", "\n"})
_REDIR_OPS = frozenset({"<", ">", ">>", "<<", "<<<", ">&", "&>", "&>>", ">|", "<>", "|&"})

# Leading shell redirection operator glued to a target (fallback for the Windows
# lexer, which does not split punctuation): "2>>out" -> "out".
_REDIR_RE = re.compile(r"^\d*(?:>>|<<|&>>|&>|>&|>\||>|<)")
# $VAR / ${VAR} references, expanded from child_env only.
_VAR_RE = re.compile(r"\$\{(\w+)\}|\$(\w+)")
# Quoted string literals in python source (best-effort, no escape decoding).
_PY_STR_RE = re.compile(r'"([^"\n]*)"|\'([^\'\n]*)\'')


def host_pathmod():
    """Return the path module matching the host: ``ntpath`` on Windows else
    ``posixpath``."""
    import sys

    return ntpath if sys.platform == "win32" else posixpath


def static_screen_enabled(env=None) -> bool:
    """True unless the sandbox FS confinement switch is explicitly set to off."""
    raw = (os.environ if env is None else env).get(_DISABLE_ENV)
    return raw is None or raw.strip().lower() not in _OFF_VALUES


def _expand(raw: str, child_env) -> "str | None":
    """Expand a leading ``~`` and ``$VAR`` / ``${VAR}`` from ``child_env`` (shell
    semantics). Returns None when a referenced variable is absent, or when a
    substituted value is a path-separator-joined list (e.g. ``$PATH``) rather than
    a single path -- both are unresolvable as one operand."""
    s = raw
    if s[:1] == "~" and (len(s) == 1 or s[1:2] in ("/", "\\")):
        home = child_env.get("HOME") or child_env.get("USERPROFILE")
        if not home:
            return None
        s = home + s[1:]
    if "$" not in s:
        return s
    out, pos = [], 0
    for m in _VAR_RE.finditer(s):
        val = child_env.get(m.group(1) or m.group(2))
        if val is None or os.pathsep in val:
            return None
        out.append(s[pos:m.start()])
        out.append(val)
        pos = m.end()
    out.append(s[pos:])
    result = "".join(out)
    return None if "$" in result else result


def _resolve(raw: str, workdir: str, child_env, pathmod, expand: bool) -> "str | None":
    """Normalized, workdir-anchored path for ``raw``, or None when empty / NUL /
    (with expansion) holding an unresolvable variable. ``expand`` is True for shell
    tokens and False for python literals (the interpreter does not expand ``~`` or
    ``$VAR`` inside a string literal, so those are ordinary relative names)."""
    if not raw or "\x00" in raw:
        return None
    s = raw
    if expand:
        s = _expand(raw, child_env)
        if s is None:
            return None
    if not pathmod.isabs(s):
        s = pathmod.join(workdir, s)
    return pathmod.normpath(s)


def _same_or_child(path: str, root: str, pathmod) -> bool:
    """Component-wise containment of ``path`` in ``root`` (pathmod-injectable).
    ``commonpath`` (not ``str.startswith``) so a sibling prefix like ``/work_evil``
    is not treated as inside ``/work``. Lexical only -- a pre-existing symlink is
    not resolved (documented time-of-check/time-of-use gap)."""
    p = pathmod.normcase(pathmod.normpath(path))
    r = pathmod.normcase(pathmod.normpath(root))
    if p == r:
        return True
    try:
        return pathmod.commonpath([p, r]) == r
    except ValueError:
        return False


def _allowed_roots(workdir: str, child_env):
    """Roots whose subtree counts as inside:

    - the session workdir;
    - the child's own temp dir (TMPDIR/TMP/TEMP -- the sandbox points these at the
      workdir) and the OS temp tree: accepted best-effort scratch. Not a strong
      boundary on a multi-tenant host, but Studio is single-operator and per-file
      size is capped elsewhere;
    - the sitecustomize remap prefixes, but ONLY while absent on the host. The shim
      remaps an absent prefix onto the workdir at runtime; an existing host dir is
      not remapped, so it stays outside. An absent dir cannot be read/written by the
      shell either, so gating on absence is safe for both python and shell."""
    roots = [workdir]
    for key in ("TMPDIR", "TMP", "TEMP"):
        tmp = child_env.get(key)
        if tmp:
            roots.append(tmp)
    try:
        roots.append(tempfile.gettempdir())
    except Exception:
        pass
    roots.extend(("/tmp", "/var/tmp"))
    for prefix in _REMAP_PREFIXES:
        try:
            if not os.path.exists(prefix):
                roots.append(prefix)
        except OSError:
            pass
    return roots


def classify_path(
    raw: str, workdir: str, child_env, *, pathmod, expand: bool = True
) -> "tuple[str, str | None]":
    """Return ``(status, resolved)`` where status is "inside" | "outside" |
    "unknown" and resolved is the normalized path (None when unknown). ``expand`` is
    True for shell tokens and False for python literals."""
    resolved = _resolve(raw, workdir, child_env, pathmod, expand)
    if resolved is None:
        return "unknown", None
    for root in _allowed_roots(workdir, child_env):
        if _same_or_child(resolved, root, pathmod):
            return "inside", resolved
    return "outside", resolved


def _pathlike(s: str, pathmod) -> bool:
    """Whether a token is worth classifying: absolute, ``~``, a variable, or a
    ``..`` traversal. A plain relative word (``echo``, ``note.txt``) is inside."""
    if not s:
        return False
    return (
        pathmod.isabs(s)
        or s[0] == "~"
        or "$" in s
        or s.startswith("..")
        or "/.." in s
        or "\\.." in s
    )


def _python_reachable(resolved: str, pathmod) -> bool:
    """Whether an outside python operand is a real host target (its parent exists).
    Mirrors the sitecustomize shim: a create-write with a missing parent is healed
    onto the workdir and a read of a missing path fails harmlessly, so neither is a
    real escape. A UNC / network parent is never stat-ed (it can hang or force an
    SMB auth) -- it is treated as reachable so it is still flagged, without touching
    the network."""
    if not resolved:
        return False
    if pathmod is ntpath and resolved.startswith(("\\\\", "//")):
        return True
    try:
        parent = os.path.dirname(resolved)
        return bool(parent) and os.path.exists(parent)
    except OSError:
        return False


def _strip_redirect(token: str) -> "str | None":
    """Strip a leading redirection operator glued to a target, returning the path
    portion, or None for an operator-only token. Fallback for the Windows lexer,
    which does not split punctuation; the posix lexer already splits these out."""
    m = _REDIR_RE.match(token)
    if not m:
        return token
    rest = token[m.end():]
    return rest or None


def _shell_tokens(command: str, pathmod) -> "list[str]":
    """Tokenize a shell command. POSIX: a punctuation-aware lexer so redirection and
    separator operators split even when glued (``cat</etc/passwd`` -> ``cat`` ``<``
    ``/etc/passwd``). Windows: ``posix=False`` so backslash paths survive, with
    surrounding quotes stripped. Malformed quoting yields no tokens (allowed)."""
    if pathmod is ntpath:
        try:
            tokens = shlex.split(command, posix=False)
        except ValueError:
            return []
        out = []
        for t in tokens:
            if len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"'):
                t = t[1:-1]
            out.append(t)
        return out
    lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;()<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        return []


def scan_shell(command: str, workdir: str, child_env, pathmod) -> "list[str]":
    """Outside operands in a shell command (argument + redirection-target tokens).
    The command word (start, and the first token after a ``;`` / ``|`` / ``&&`` /
    ``||`` separator) is exempt -- it is an executable resolved through the sandbox
    PATH, not a file operand. A nested ``bash -c '...'`` payload is not recursed into
    (documented best-effort gap)."""
    tokens = _shell_tokens(command, pathmod)
    out = []
    expect_command = True
    expect_target = False
    for tok in tokens:
        if tok in _SEPARATORS:
            expect_command = True
            expect_target = False
            continue
        if tok in _REDIR_OPS:
            expect_target = True
            continue
        path = _strip_redirect(tok)
        had_glued_redirect = path != tok
        if path is None:
            expect_target = True
            continue
        is_target = expect_target or had_glued_redirect
        expect_target = False
        if expect_command and not is_target:
            expect_command = False  # the executable itself, resolved via PATH
            continue
        if path in _DEV_ALLOWED or path.startswith("/dev/fd/"):
            continue
        if "://" in path or not _pathlike(path, pathmod):
            continue
        status, _ = classify_path(path, workdir, child_env, pathmod=pathmod, expand=True)
        if status == "outside":
            out.append(path)
    return out


def scan_python(code: str, workdir: str, child_env, pathmod) -> "list[str]":
    """Outside operands among the quoted string literals in python source. Literals
    are not env-expanded (the interpreter does not expand ``~`` / ``$VAR`` inside a
    string). A literal whose parent does not exist on the host is left to the
    sitecustomize write-remap shim."""
    out = []
    for m in _PY_STR_RE.finditer(code):
        lit = m.group(1) if m.group(1) is not None else m.group(2)
        if not lit or "://" in lit or not _pathlike(lit, pathmod):
            continue
        status, resolved = classify_path(lit, workdir, child_env, pathmod=pathmod, expand=False)
        if status == "outside" and _python_reachable(resolved, pathmod):
            out.append(lit)
    return out


def check_static_fs(kind: str, source: str, workdir: str, child_env, pathmod) -> "str | None":
    """Return a one-line rejection when ``source`` provably reads or writes outside
    the workdir, else None. Any analyzer error is swallowed (returns None) so a
    screening bug never blocks legitimate work."""
    try:
        if kind == "python":
            outside = scan_python(source, workdir, child_env, pathmod)
        elif kind == "shell":
            outside = scan_shell(source, workdir, child_env, pathmod)
        else:
            return None
    except Exception:
        return None
    if outside:
        return (
            f"Blocked for safety: {outside[0]!r} is outside the sandbox working "
            f"directory; read and write only inside the working directory "
            f"(best-effort static check, not a full sandbox)."
        )
    return None
