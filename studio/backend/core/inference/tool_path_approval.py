# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Approval decisions for file access that leaves the session sandbox.

Split out of ``tools.py`` because it is a subsystem rather than a handful of helpers: the read and
write silent roots, the shared ``_path_needs_approval`` predicate, and the two operand scanners that
decide which tokens of a terminal command or which arguments of a python snippet are paths at all.

The import between this module and ``tools`` runs both ways, so both sides bind the MODULE rather
than its names and resolve attributes at call time. That is what keeps the cycle harmless whichever
module is imported first.
"""

import ast
import fnmatch
import functools
import os
import re
import shlex
import sys
import time
import urllib.parse

# Bound at the bottom of `tools`, not imported: the modules import each other.
_BORROWED = (
    "_ARCHIVE_CTOR_NAMES",
    "_AUTO_SAFE_WRAPPERS",
    "_AUTO_UNSAFE_PY_WRITE_METHODS",
    "_GLOB_META_RE",
    "_MAX_PATH_SCAN_CHARS",
    "_PATH_CTORS",
    "_REDIR_PREFIX_RE",
    "_SANDBOX_SITE_DIR",
    "_WIN_DRIVE_RE",
    "_WRAPPER_VALUE_FLAGS_BY_CMD",
    "_folded_path",
    "_glob_token_sensitive",
    "_has_kwarg_splat",
    "_legacy_sandbox_root",
    "_mode_arg_writes",
    "_parse_python",
    "_posix_join",
    "_references_sensitive_path",
    "_token_command_base",
    "_tree_nodes",
    "current_account_id",
    "sandbox_root",
)


def _bind(namespace) -> None:
    """Bind the borrowed names from `tools`, once, at the end of its import.

    Two of the tables here EXTEND a table that lives in `tools`, and a module-level union would run
    before that module finished. They are completed here instead, along with the union derived from
    them, so the tables are whole before the first classification.
    """
    globals().update({name: namespace[name] for name in _BORROWED if name in namespace})
    global _PY_PATH_WRITE_CALLS, _PY_PATH_ARCHIVE_CTORS, _PY_ALIASABLE_PATH_CALLS
    _PY_PATH_WRITE_CALLS = _PY_PATH_WRITE_CALLS | frozenset(
        namespace.get("_AUTO_UNSAFE_PY_WRITE_METHODS", ())
    )
    _PY_PATH_ARCHIVE_CTORS = _PY_PATH_ARCHIVE_CTORS | frozenset(
        namespace.get("_ARCHIVE_CTOR_NAMES", ())
    )
    _PY_ALIASABLE_PATH_CALLS = (
        frozenset({"open", "fdopen"})
        | _PY_PATH_READ_CALLS
        | _PY_PATH_WRITE_CALLS
        | _PY_PATH_DEST_SECOND_CALLS
        | _PY_PATH_SERIALIZE_CALLS
        | _PY_PATH_CONTENT_FIRST_CALLS
        | _PY_PATH_ARCHIVE_CTORS
        | _PY_PATH_SUBPROCESS_CALLS
    )


# The owner's tool child runs unconfined, so an absolute path reaches the real filesystem;
# relative paths stay in the workdir, and absolute paths outside these roots ask in auto mode.
_SYSTEM_READ_SILENT_ROOTS = (
    # Credential checks run first, so /etc/shadow still asks despite /etc being here.
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib32",
    "/lib64",
    "/opt",
    "/snap",
    "/nix",
    "/etc",
    "/var/lib",
    "/proc",
    "/sys",
    "/dev",
    # Mirrors tool_confinement._SYSTEM_READ_ROOTS; secret subdirs are caught by the credential check.
    "/run",
    # On macOS /var/run is the real location and /etc/resolv.conf links into it.
    "/var/run",
    "/System",
    "/Library",
    "/Applications",
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "C:\\ProgramData",
)


# Carved out of the system roots: user-owned tmpfs holds user data, not machine state.
_WRITABLE_RUNTIME_SUBTREES = tuple(
    os.path.normcase(part)
    # `/dev/fd/<n>` reaches whatever the descriptor was opened on, like `/proc/self/fd/<n>`.
    for part in ("/dev/shm", "/dev/mqueue", "/dev/fd", "/run/user", "/run/lock")
)


_WRITE_SILENT_DEVICE_NODES = (
    "/dev/null",
    "/dev/zero",
    "/dev/stdout",
    "/dev/stderr",
    "/dev/tty",
    "/dev/full",
)


# A stale entry only costs or spares one prompt, so a short TTL beats invalidation.
_SILENT_ROOT_TTL_S = 60.0


_silent_roots_cache = None


def _normalized_fs_text(text: str) -> str:
    """Fold a path to the form root containment compares against: `~` expanded, separators
    canonical, `.`/`..` collapsed, case-folded where the platform is case-insensitive."""
    if os.sep != "/":
        text = text.replace("/", os.sep)
    if text[:1] == "~":
        try:
            text = os.path.expanduser(text)
        except Exception:  # noqa: BLE001 - a broken HOME must not break classification
            pass
    return os.path.normcase(os.path.normpath(text)).rstrip(os.sep) or os.sep


def _silent_root_list(producers) -> "tuple[str, ...]":
    """Normalize each producer's paths, skipping any that raises: a root that cannot be resolved
    (a retired account, an unset studio home) must narrow the allowlist, never crash the scan."""
    roots: "list[str]" = []
    for producer in producers:
        try:
            value = producer()
        except Exception:  # noqa: BLE001 - every root is best effort
            continue
        if value is None:
            continue
        items = value if isinstance(value, (list, tuple, set)) else (value,)
        for item in items:
            if not item:
                continue
            try:
                root = _normalized_fs_text(str(item))
            except Exception:  # noqa: BLE001
                continue
            # "/" would make every path silent, so a root that folds to the filesystem root is dropped.
            if root and root != os.sep and root not in roots:
                roots.append(root)
    return tuple(roots)


def _excluding_ancestors_of_studio_home(roots) -> "tuple[str, ...]":
    """Drop any root that is the studio home or contains it.

    ``studio.db`` (chat history, provider and MCP configuration) and ``auth/`` live directly in that
    directory, so a root reaching it hands a tool all of Studio's own state.
    """
    from utils.paths.storage_roots import studio_root

    try:
        home = _normalized_fs_text(str(studio_root()))
    except Exception:  # noqa: BLE001 - an unresolvable home means nothing to protect against here
        return tuple(roots)
    return tuple(root for root in roots if not (home == root or home.startswith(root + os.sep)))


def _build_silent_roots() -> "tuple[tuple[str, ...], tuple[str, ...]]":
    """(read-silent, write-silent) roots. Write-silent is the narrower set: where tool output
    ordinarily lands. Read-silent adds installed software and the model libraries the user pointed
    Studio at, which a tool reads all the time and which carry no user documents."""
    from utils.paths.storage_roots import (
        cache_root,
        project_workspaces_root,
        shared_project_workspaces_root,
        shared_tmp_root,
        tmp_root,
        well_known_model_dirs,
    )

    # Only output subdirectories, never the studio home: it holds studio.db and auth/.
    write_roots = _silent_root_list(
        (
            sandbox_root,
            _legacy_sandbox_root,
            tmp_root,
            # Not the shared bases: they are ancestors of every account's private data.
            project_workspaces_root,
            cache_root,
            # Studio downloads models here; credential files inside are caught by the sensitive-path check.
            lambda: _hf_cache_dirs(),
            lambda: _WRITE_SILENT_DEVICE_NODES,
        )
    )
    # Producers fall back to the studio home when unconfigured; drop it and its ancestors.
    write_roots = _excluding_ancestors_of_studio_home(write_roots)
    read_roots = write_roots + _excluding_ancestors_of_studio_home(
        _silent_root_list(
            (
                lambda: _SYSTEM_READ_SILENT_ROOTS,
                lambda: (
                    sys.prefix,
                    sys.base_prefix,
                    sys.exec_prefix,
                    getattr(sys, "base_exec_prefix", ""),
                    os.path.dirname(sys.executable),
                    os.environ.get("VIRTUAL_ENV", ""),
                    _SANDBOX_SITE_DIR,
                ),
                _scan_folder_roots,
                well_known_model_dirs,
            )
        )
    )
    return read_roots, write_roots


def _hf_cache_dirs() -> "tuple[str, ...]":
    """The HF cache locations, including one the owner configured in the UI.

    Skipped entirely when the database does not exist yet, for the same reason the scan folders are:
    the configured value is read through `get_app_setting`, which opens `studio.db`, and a
    classification must not be what creates it. The environment and default locations still resolve,
    so a first run keeps the caches it actually has.
    """
    from utils.hf_cache_settings import known_hf_cache_homes, known_hf_hub_caches

    if not _studio_db_exists():
        from utils.hf_cache_settings import get_hf_cache_paths
        try:
            paths = get_hf_cache_paths()
        except Exception:  # noqa: BLE001 - best effort, as every root here is
            return ()
        return tuple(str(p) for p in (paths.cache_home, paths.hub_cache, paths.xet_cache) if p)
    return tuple(str(p) for p in (*known_hf_cache_homes(), *known_hf_hub_caches()))


_studio_db_path_cache: "tuple | None" = None


def _studio_db_path_for(identity) -> "str | None":
    """`studio.db`'s path, memoized on the account and environment the roots cache is keyed on.

    `studio_db_path()` re-resolves the studio root on every call, which measured at ~28us here and
    was most of the cost of VALIDATING an already cached root set. The memo key is the same one the
    roots cache uses, so a changed home still re-resolves; only the repeated lookup is saved.
    """
    global _studio_db_path_cache
    cached = _studio_db_path_cache
    if cached is not None and cached[0] == identity:
        return cached[1]
    from storage.studio_db import studio_db_path

    try:
        path = str(studio_db_path())
    except Exception:  # noqa: BLE001 - an unresolvable path is no database
        path = None
    _studio_db_path_cache = (identity, path)
    return path


def _studio_db_revision(identity = ()) -> int:
    """The database's modification time, or 0 when there is none. Changes whenever a root that is
    stored in it does.

    The write-ahead log counts too: the server runs a WAL keeper, so a committed change to a scan
    folder or a cache setting lands in `studio.db-wal` and leaves the main file's mtime alone. Read
    on its own, a revoked folder stayed read-silent until the TTL expired.
    """
    path = _studio_db_path_for(identity)
    if path is None:
        return 0
    try:
        revision = os.stat(path).st_mtime_ns
    except Exception:  # noqa: BLE001 - no database is a stable revision of its own
        return 0
    try:
        wal = os.stat(f"{path}-wal")
        # Size as well as mtime: a commit within the same mtime granularity still grows the log.
        revision ^= wal.st_mtime_ns ^ wal.st_size
    except Exception:  # noqa: BLE001 - no WAL is simply no extra revision
        pass
    return revision


def _studio_db_exists() -> bool:
    """Whether `studio.db` is already there.

    Opening it CREATES it and initialises 27 tables, so any root that reads a stored setting has to
    ask this first: on a first run the very first tool call would otherwise create the database as a
    side effect of deciding whether to ask about a path.
    """
    from storage.studio_db import studio_db_path
    try:
        return studio_db_path().exists()
    except Exception:  # noqa: BLE001 - an unresolvable path is no stored setting either
        return False


def _scan_folder_roots() -> "tuple[str, ...]":
    """Model folders the owner added in the UI. Read from the same table the model browser uses.

    Only when the database already EXISTS. Opening it creates the file and initialises the schema,
    and a classification is not a reason for that to happen: on a first run the very first tool call
    would have created `studio.db` as a side effect of deciding whether to ask about a path. No
    database means no folders were ever registered, which is the same answer an empty table gives.
    """
    from storage.studio_db import list_scan_folders

    if not _studio_db_exists():
        return ()
    return tuple(str(row.get("path") or "") for row in list_scan_folders())


# Env vars that relocate a root are part of the cache key so a repoint takes effect at once.
_SILENT_ROOT_ENV_KEYS = (
    "UNSLOTH_STUDIO_HOME",
    "STUDIO_HOME",
    "UNSLOTH_STUDIO_PROJECTS_HOME",
    "UNSLOTH_STUDIO_DOCUMENTS_HOME",
    "OLLAMA_MODELS",
    "UNSLOTH_STUDIO_SANDBOX_HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "XDG_CACHE_HOME",
    "VIRTUAL_ENV",
)


def _silent_roots() -> "tuple[tuple[str, ...], tuple[str, ...]]":
    global _silent_roots_cache
    try:
        account = current_account_id() or ""
    except Exception:  # noqa: BLE001
        account = ""
    # The db mtime too: scan folders and the cache root live in it.
    identity = (account, tuple(os.environ.get(k) for k in _SILENT_ROOT_ENV_KEYS))
    account = identity + (_studio_db_revision(identity),)
    now = time.monotonic()
    cached = _silent_roots_cache
    if cached is not None and cached[1] == account and now - cached[0] < _SILENT_ROOT_TTL_S:
        return cached[2], cached[3]
    try:
        read_roots, write_roots = _build_silent_roots()
    except Exception:  # noqa: BLE001 - no allowlist is the fail-closed direction (more prompts)
        read_roots, write_roots = (), ()
    _silent_roots_cache = (now, account, read_roots, write_roots)
    return read_roots, write_roots


def _looks_absolute(text: str) -> bool:
    """True for a path that reaches the real filesystem rather than the session workdir: POSIX
    absolute, `~`-rooted, a Windows drive path, or a UNC share. Windows spellings count on every
    platform, since misjudging one only costs a prompt."""
    if not text:
        return False
    if text[0] == "/":
        return True
    if text[:5].lower() == "file:":
        uri_path = _file_uri_path(text)
        return bool(uri_path) and _looks_absolute(uri_path)
    if text[0] == "~":
        # A plain `~` is the sandbox (child HOME is the workdir); `~alice` stays absolute.
        return not (len(text) == 1 or text[1] in "/\\")
    # A leading backslash is root-relative on Windows, not sandbox-relative.
    if text[0] == "\\":
        return True
    # `C:notes.txt` resolves against drive C's current directory, not the session workdir.
    return bool(_WIN_DRIVE_RE.match(text) or _WIN_DRIVE_RELATIVE_RE.match(text))


# Anchored and single-colon so `${p:0:3}` is not read as a path on drive P.
_WIN_DRIVE_RELATIVE_RE = re.compile(r"^[A-Za-z]:(?![\\/:])[^:\s]+$")


# Kernel links under /proc that reach outside it; shell PID spellings name the same links.
_PROC_ID_RE = r"(?:self|thread-self|\d+|\$\$|\$\{?BASHPID\}?|\$\{?PPID\}?)"
_PROC_MAGIC_LINK_RE = re.compile(
    rf"/proc/{_PROC_ID_RE}/(?:task/{_PROC_ID_RE}/)?(?:root|cwd|fd)(?:/|$)"
)


# Local `file:` URI spellings; a non-localhost host is remote, not a path.
_FILE_URI_RE = re.compile(r"^file://(?:localhost)?(?=/)|^file:(?=/)", re.IGNORECASE)


def _contained_in(candidate: str, roots) -> bool:
    """Component-boundary containment, never a raw prefix test."""
    return any(candidate == root or candidate.startswith(root + os.sep) for root in roots)


def _resolved_fs_text(text: str) -> str:
    """*text* with its symlinked components resolved, folded the way the roots are."""
    try:
        return _normalized_fs_text(os.path.realpath(text))
    except Exception:  # noqa: BLE001 - an unresolvable path keeps the lexical answer
        return text


def _file_uri_path(text: str) -> "str | None":
    """The filesystem path a local `file:` URI names, or None if *text* is not one.

    `sqlite3.connect("file:/media/alice/private.db?mode=ro", uri = True)` opens the same database the
    bare path does, but the scheme made it read as a relative name and the read went silent. The
    query and fragment are the URI's own, not part of the path, and `%XX` is decoded because the OS
    sees the decoded form.
    """
    if not _FILE_URI_RE.match(text):
        return None
    path = _FILE_URI_RE.sub("", text, count = 1)
    for separator in ("?", "#"):
        cut = path.find(separator)
        if cut != -1:
            path = path[:cut]
    if "%" in path:
        try:
            path = urllib.parse.unquote(path)
        except Exception:  # noqa: BLE001 - a malformed escape must not break classification
            pass
    return path or None


def _names_a_managed_account_subtree(candidate: str) -> bool:
    """Whether *candidate* descends into a per-account directory of a managed install."""
    parts = candidate.replace("\\", "/").lower().split("/")
    return any(part == "accounts" and index + 1 < len(parts) for index, part in enumerate(parts))


def _path_needs_approval(text, *, writing: bool = False) -> bool:
    """True when reading (or, with ``writing``, creating/overwriting) this path leaves the sandbox
    for the user's own filesystem.

    Order matters: the credential checks run first, so the allowlist below can never turn
    ``/etc/shadow`` or ``~/.ssh/id_rsa`` into a silent read just because ``/etc`` is read-silent.
    A relative path stays silent -- it resolves inside the per-session workdir.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    text = text.strip()
    if len(text) > _MAX_PATH_SCAN_CHARS:
        return True
    # \x02 marks a pathlib .parent escaping its root (see _folded_path): ask.
    if "\x02" in text:
        return True
    if "\x00" in text:
        return False
    uri_path = _file_uri_path(text)
    if uri_path is not None:
        return _path_needs_approval(uri_path, writing = writing)
    if not _looks_absolute(text):
        return bool(_references_sensitive_path(text) or _glob_token_sensitive(text))
    try:
        candidate = _normalized_fs_text(text)
    except Exception:  # noqa: BLE001
        return True
    # The filesystem root itself: listing it reveals only top-level names. Exact match only.
    if not writing and candidate == os.sep:
        return False
    # Kernel magic symlinks: `/proc/self/root/home/x` opens `/home/x` while looking like /proc.
    if _PROC_MAGIC_LINK_RE.match(candidate.replace(os.sep, "/")):
        return True
    # User-writable tmpfs under a system root; checked before containment.
    if any(
        candidate == root or candidate.startswith(root + os.sep)
        for root in _WRITABLE_RUNTIME_SUBTREES
    ):
        return True
    read_roots, write_roots = _silent_roots()
    roots = write_roots if writing else read_roots
    if not _contained_in(candidate, roots):
        return True
    # Shared project and tmp bases hold every account's subtree; never silent.
    if _names_a_managed_account_subtree(candidate):
        return True
    # A symlink inside a silent root can resolve outside it. Resolved only on the about-to-allow
    # path; the TOCTOU gap is the OS sandbox's to close.
    resolved = _resolved_fs_text(candidate)
    # A root that is itself a symlink (macOS /tmp) would otherwise reject everything under it.
    if resolved != candidate and not _contained_in(resolved, roots):
        if not _contained_in(resolved, tuple(_resolved_fs_text(r) for r in roots)):
            return True
    # Only a credential under a silent root still asks; check the resolved target too
    # (`/models/public -> $HF_HOME/token`).
    if _credential_under_silent_root(candidate) or (
        resolved != candidate and _credential_under_silent_root(resolved)
    ):
        return True
    if resolved != candidate and _references_sensitive_path(resolved):
        return True
    return bool(_references_sensitive_path(text) or _glob_token_sensitive(text))


# Name-based so a relocated cache (HF_HOME=/media/hf-cache) still hides its token store.
_CREDENTIAL_BASENAMES = frozenset(
    {
        "token",
        "stored_tokens",
        "credentials",
        "auth.db",
        ".netrc",
        "netrc",
        ".git-credentials",
        ".pypirc",
        ".npmrc",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "id_dsa",
    }
)


_CREDENTIAL_PATH_SUFFIXES = (
    "/etc/gshadow",
    "/etc/gshadow-",
    "/etc/krb5.keytab",
    "/etc/master.passwd",
    "/etc/security/opasswd",
    "/etc/ipsec.secrets",
)


def _credential_under_silent_root(candidate: str) -> bool:
    """True for a secret sitting inside an otherwise silent root, so the allowlist cannot expose it."""
    if os.path.basename(candidate) in _CREDENTIAL_BASENAMES:
        return True
    # Separator-unified so Windows hosts match POSIX paths too.
    unified = candidate.replace("\\", "/")
    if any(unified.endswith(suffix) for suffix in _CREDENTIAL_PATH_SUFFIXES):
        return True
    return "/auth/" in unified + "/"


# Read commands; deliberately narrow: only names whose operands are unambiguously paths.
# `cd` to an absolute path re-points later relative operands.
_PATH_READ_COMMANDS = frozenset(
    """
    cat zcat bzcat xzcat lzcat lz4cat zstdcat iconv test [ [[ tac head tail less more wc stat
    file od xxd hexdump strings base64 b64encode md5 md5sum sha1sum sha224sum sha256sum
    sha384sum sha512sum shasum cksum sum cut sort uniq nl rev column paste join comm expand
    unexpand fold fmt jq yq diff cmp grep egrep fgrep rg ag ack ls dir du tree find fd readlink
    realpath awk gawk mawk sed openssl xmllint csvlook type get-content gc tar 7z unrar cd pushd
    """.split()
)


_PATH_WRITE_COMMANDS = frozenset(
    {
        # The first operand is a mode/owner, skipped through `_PATH_ARG_SKIP`.
        "chmod",
        "chown",
        "chgrp",
        "tee",
        "touch",
        "mkdir",
        "truncate",
        "shred",
        "zip",
        "gzip",
        "gunzip",
        "bzip2",
        "bunzip2",
        "xz",
        "unxz",
        "zstd",
    }
)
# With -c/--stdout the operand is only read.
_STDOUT_COMPRESSORS = frozenset({"gzip", "gunzip", "bzip2", "bunzip2", "xz", "unxz", "zstd"})
_STDOUT_FLAGS = frozenset({"-c", "--stdout", "--to-stdout"})


_PATH_DEST_LAST_COMMANDS = frozenset({"cp", "mv", "install", "ln", "rsync"})


# First positional is the archive written; later ones are read sources (`-m` makes them writes).
_PATH_DEST_FIRST_COMMANDS = frozenset({"zip"})


# Source checked as a write: `mv` removes it, and `ln` hands the sandbox a name that writes to it.
_PATH_SOURCE_MUTATING_COMMANDS = frozenset({"mv", "ln"})


# Interpreters loading their operand files, treated as reads. `awk`/`jq` are absent: their
# first positional is a program.
_PATH_SCRIPT_COMMANDS = frozenset(
    """
    python python2 python3 py perl bash sh zsh ksh dash fish csh tcsh source ruby node deno bun
    php lua luajit julia rscript duckdb curl wget date unzip
    """.split()
)


# Paths only via flags; positionals here are subcommands, filters or sources.
_PATH_FLAG_ONLY_COMMANDS = frozenset("git make jq gcc g++ cc c++ clang clang++".split())


# First positional is a program or pattern, not a file; the value is how many to skip.
_PATH_ARG_SKIP = {
    "chmod": 1,
    "chown": 1,
    "chgrp": 1,
    "sed": 1,
    "awk": 1,
    "gawk": 1,
    "mawk": 1,
    "jq": 1,
    "yq": 1,
    "grep": 1,
    "egrep": 1,
    "fgrep": 1,
    "rg": 1,
    "ag": 1,
    "ack": 1,
    "tar": 1,
    "openssl": 1,
    "xargs": 1,
}


# GNU tar recognises the legacy option word only as the very first argument.
_PATH_SKIP_FIRST_ARG_ONLY = frozenset({"tar"})


# Long spellings of archive write modes, including --delete and --concatenate.
_ARCHIVE_CREATE_LONG_FLAGS = frozenset(
    {"--create", "--append", "--update", "--delete", "--concatenate", "--catenate"}
)


_ARCHIVE_WRITE_SHORT_MODES = ("c", "r", "u", "A")
# `7z`: the first word is the command; tar-style mode letters mean nothing here.
_SEVENZIP_WRITE_COMMANDS = frozenset({"a", "u", "d", "rn"})


# These flags supply the pattern, so no positional is skipped for one.
_PATTERN_SUPPLYING_FLAGS = {
    "grep": {"-f", "--file", "-e", "--regexp"},
    "egrep": {"-f", "--file", "-e", "--regexp"},
    "fgrep": {"-f", "--file", "-e", "--regexp"},
    # `rg --files` takes no pattern.
    "rg": {"-f", "--file", "-e", "--regexp", "--files"},
    "ag": {"-f", "--file", "-e"},
    "ack": {"-f", "--file"},
    "sed": {"-f", "--file", "-e", "--expression"},
    "awk": {"-f", "--file"},
    "gawk": {"-f", "--file"},
    "mawk": {"-f", "--file"},
}


# Module-level `open()` functions taking the path first (unlike `Path(p).open()`), including
# tarfile, builtins and PIL Image.
_PY_MODULE_OPEN_RECEIVERS = frozenset(
    """
    io os posix gzip bz2 lzma codecs tokenize dbm shelve wave tarfile builtins __builtin__ PIL
    Image
    """.split()
)


# Module receivers put paths in the arguments: `os.rename(src, dst)` vs `Path(src).rename(dst)`.
_PY_MODULE_PATH_RECEIVERS = _PY_MODULE_OPEN_RECEIVERS | frozenset(
    {
        "shutil",
        "pathlib",
        "zipfile",
        "json",
        "pickle",
        "numpy",
        "np",
        "torch",
        "joblib",
        "cv2",
        "pandas",
        "pd",
    }
)


# `test`/`[` touch the filesystem only through file operators; other operands are strings.
_TEST_COMMANDS = frozenset({"test", "[", "[["})
_TEST_FILE_UNARY_FLAGS = frozenset(
    """
    -a -b -c -d -e -f -g -h -k -p -r -s -u -w -x -G -L -N -O -S
    """.split()
)
_TEST_FILE_BINARY_OPS = frozenset({"-nt", "-ot", "-ef"})


def _test_command_operands(args) -> "list[str]":
    """The tokens `test`/`[` actually stat: the operand of a unary file test and both sides of
    `-nt`/`-ot`/`-ef`."""
    paths: "list[str]" = []
    for index, arg in enumerate(args):
        if arg in _TEST_FILE_UNARY_FLAGS:
            if index + 1 < len(args):
                paths.append(args[index + 1])
        elif arg in _TEST_FILE_BINARY_OPS:
            if index:
                paths.append(args[index - 1])
            if index + 1 < len(args):
                paths.append(args[index + 1])
    return paths


_PATH_FORWARDING_COMMANDS = frozenset({"xargs", "parallel"})


@functools.lru_cache(maxsize = 1)
def _classified_terminal_commands() -> frozenset:
    """Every command this scan can reason about, path-bearing or not.

    Used by the subprocess fallback to tell "the child accesses no path" from "the child is a name
    nothing here models". `_AUTO_SAFE_TERMINAL_COMMANDS` is the second half of that: `echo` and
    `printf` take no path operand, and treating their arguments as paths asked for approval on a
    child that the identical terminal command runs silently.
    """
    return frozenset(
        _PATH_READ_COMMANDS
        | _PATH_WRITE_COMMANDS
        | _PATH_DEST_LAST_COMMANDS
        | _PATH_SCRIPT_COMMANDS
        | _PATH_FLAG_ONLY_COMMANDS
        | _PATH_ARCHIVE_COMMANDS
        | _PATH_FORWARDING_COMMANDS
        | frozenset(_PATH_FLAG_SPECS)
        | frozenset(_tools._AUTO_SAFE_TERMINAL_COMMANDS)
    )


# Flag value kinds: read/write path, "skip" for data (`cut -d /`), "archive" per mode. An
# unlisted flag does not consume the next token, so its path is still seen as a positional.
_PATH_FLAG_SPECS = {
    "cp": {"-t": "write", "--target-directory": "write", "-S": "skip", "--suffix": "skip"},
    "mv": {"-t": "write", "--target-directory": "write", "-S": "skip", "--suffix": "skip"},
    "iconv": {
        "-o": "write",
        "--output": "write",
        "-f": "skip",
        "--from-code": "skip",
        "-t": "skip",
        "--to-code": "skip",
    },
    # `-f` and `-C` select what make reads and runs; bare positionals are targets.
    "make": {
        "-f": "read",
        "--file": "read",
        "--makefile": "read",
        "-C": "write",
        "--directory": "write",
        "-j": "skip",
        "--jobs": "skip",
        "-l": "skip",
    },
    "gcc": {"-o": "write"},
    "g++": {"-o": "write"},
    "cc": {"-o": "write"},
    "c++": {"-o": "write"},
    "clang": {"-o": "write"},
    "clang++": {"-o": "write"},
    "install": {
        "-t": "write",
        "--target-directory": "write",
        "-m": "skip",
        "-g": "skip",
        "-o": "skip",
    },
    "ln": {"-t": "write", "--target-directory": "write", "-S": "skip", "--suffix": "skip"},
    "rsync": {"--exclude": "skip", "--exclude-from": "read", "--files-from": "read"},
    "sort": {
        "-o": "write",
        "--output": "write",
        "--files0-from": "read",
        "--random-source": "read",
        "-t": "skip",
        "--field-separator": "skip",
        "-k": "skip",
        "--key": "skip",
        "-T": "skip",
        "--temporary-directory": "skip",
        "-S": "skip",
        "--buffer-size": "skip",
    },
    "grep": {
        "-f": "read",
        "--file": "read",
        "--exclude-from": "read",
        "-e": "skip",
        "--regexp": "skip",
        "--include": "skip",
        "--exclude": "skip",
        "--exclude-dir": "skip",
        "-m": "skip",
        "-A": "skip",
        "-B": "skip",
        "-C": "skip",
        "-d": "skip",
        "-D": "skip",
    },
    "tar": {
        "-f": "archive",
        "--file": "archive",
        # The snapshot is written regardless of create or extract.
        "-g": "write",
        "--listed-incremental": "write",
        "-C": "extract_dir",
        "--directory": "extract_dir",
        "-T": "read",
        "--files-from": "read",
        "--add-file": "read",
        "-X": "read",
        "--exclude-from": "read",
        "--exclude": "skip",
    },
    # `--from-file` is read; non-path value options are listed so they are stepped over.
    "diff": {
        "--from-file": "read",
        "--to-file": "read",
        "-X": "read",
        "--exclude-from": "read",
        "-S": "read",
        "--starting-file": "read",
        "-x": "skip",
        "--exclude": "skip",
        "-D": "skip",
        "--ifdef": "skip",
        "--label": "skip",
        "-W": "skip",
        "--width": "skip",
        "--tabsize": "skip",
        "--horizon-lines": "skip",
        "--line-format": "skip",
        "-C": "skip",
        "--context": "skip",
        "-U": "skip",
        "--unified": "skip",
    },
    # `git -C <path>` runs the command in that directory.
    "git": {
        "-C": "read",
        "--git-dir": "read",
        "--work-tree": "read",
        # `git archive` options; no other git subcommand takes these names.
        "--add-file": "read",
        "--output": "write",
        "-o": "write",
        "--separate-git-dir": "write",
        "-c": "skip",
        "--exec-path": "skip",
        "--namespace": "skip",
    },
    # Only the `file:` scheme reaches the gate; http(s) URLs are not paths.
    "curl": {
        "-K": "read",
        "--config": "read",
        "-o": "write",
        "--output": "write",
        # `-O` contributes no operand, so the download dir is the only signal.
        "--output-dir": "write",
        "-O": "skip",
        "-d": "skip",
        "--data": "skip",
        "-H": "skip",
        "--header": "skip",
        "-X": "skip",
        "--request": "skip",
        "-u": "skip",
        "--user": "skip",
        "-A": "skip",
        "--user-agent": "skip",
        "-e": "skip",
        "--referer": "skip",
        "--max-time": "skip",
        "--connect-timeout": "skip",
    },
    "wget": {
        "-O": "write",
        "--output-document": "write",
        "-P": "write",
        "--directory-prefix": "write",
        "-o": "write",
        "--output-file": "write",
        "-a": "write",
        "--append-output": "write",
        "--header": "skip",
    },
    "zip": {"-x": "skip", "-i": "skip"},
    # `-d` is the extraction target; the archive itself is a positional read.
    "unzip": {"-d": "write", "-x": "skip", "-O": "skip", "-P": "skip"},
    "cut": {
        "-d": "skip",
        "--delimiter": "skip",
        "-f": "skip",
        "--fields": "skip",
        "-c": "skip",
        "--characters": "skip",
        "-b": "skip",
        "--bytes": "skip",
        "--output-delimiter": "skip",
    },
    "awk": {"-f": "read", "--file": "read", "-v": "skip", "--assign": "skip", "-F": "skip"},
    "sed": {"-f": "read", "--file": "read", "-e": "skip", "--expression": "skip", "-l": "skip"},
    "jq": {
        "-f": "read",
        "--from-file": "read",
        # `-L` lets the filter `include` modules from that directory.
        "-L": "read",
        "--library-path": "read",
        "--arg": "skip",
        "--argjson": "skip",
        "--indent": "skip",
    },
    "find": {
        "-name": "skip",
        "-iname": "skip",
        "-path": "skip",
        "-regex": "skip",
        "-newer": "read",
    },
    "fd": {
        "--base-directory": "read",
        "--search-path": "read",
        "--ignore-file": "read",
        "--exclude": "skip",
    },
    "column": {"-s": "skip", "--separator": "skip", "-o": "skip", "--output-separator": "skip"},
    "paste": {"-d": "skip", "--delimiters": "skip"},
    "join": {"-t": "skip", "-1": "skip", "-2": "skip", "-e": "skip", "-o": "skip"},
    "head": {"-n": "skip", "--lines": "skip", "-c": "skip", "--bytes": "skip"},
    "tail": {"-n": "skip", "--lines": "skip", "-c": "skip", "--bytes": "skip"},
    "openssl": {"-in": "read", "-out": "write"},
    "du": {"--exclude": "skip"},
    # `-f` reads dates from a file (bad lines are echoed) and `-r` reads its mtime; `-d` is a string.
    "date": {
        "-f": "read",
        "--file": "read",
        "-r": "read",
        "--reference": "read",
        "-d": "skip",
        "--date": "skip",
        "-s": "skip",
    },
    "tree": {"-o": "write", "-P": "skip", "-I": "skip"},
    "cd": {},
}


# ripgrep's own path options, on top of the grep spec it inherits below.
_RG_EXTRA_FLAG_SPEC = {
    "--ignore-file": "read",
    "--pre": "skip",
    "--type-add": "skip",
    "--colors": "skip",
}


for _alias, _base in (
    ("egrep", "grep"),
    ("fgrep", "grep"),
    ("rg", "grep"),
    ("ag", "grep"),
    ("ack", "grep"),
    ("gawk", "awk"),
    ("mawk", "awk"),
    ("yq", "jq"),
    ("7z", "zip"),
    ("pushd", "cd"),
):
    _PATH_FLAG_SPECS[_alias] = _PATH_FLAG_SPECS[_base]
_PATH_FLAG_SPECS["rg"] = {**_PATH_FLAG_SPECS["rg"], **_RG_EXTRA_FLAG_SPEC}

_PATH_ARCHIVE_COMMANDS = frozenset({"tar", "zip", "7z"})


# Wrapper flags taking a separate value; per-command sets override the shared fallback.
_WRAPPER_VALUE_FLAGS = frozenset(
    {"-n", "-u", "--unset", "-S", "--signal", "-k", "--kill-after", "--chdir", "-C"}
)


_WRAPPER_LEADING_VALUE_COMMANDS = frozenset({"timeout", "nice", "ionice", "stdbuf"})


_WRAPPER_LEADING_DURATION_RE = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")


_PATH_HINT_RE = re.compile(r"[/~\\:]")


_SED_INPLACE_FLAGS = ("-i", "--in-place")
# A short option that consumes the rest of its token as a value, so `perl -Ilib` is not an `-i`.
_SHORT_OPTION_VALUE_CHARS = "eflI"


def _clusters_inplace(arg: str) -> bool:
    """`sed -ni` edits in place just as `sed -i` does: short options cluster into one token."""
    if not arg.startswith("-") or arg.startswith("--"):
        return False
    for char in arg[1:]:
        if char == "i":
            return True
        if char in _SHORT_OPTION_VALUE_CHARS:
            break
    return False


_REDIR_WRITE_RE = re.compile(r"^\d*(?:>>?\|?|&>>?)$")


# A single `<` only: `<<` and `<<<` take a delimiter or literal data, never a file.
_REDIR_READ_RE = re.compile(r"^\d*<$")


_REDIR_HEREDOC_RE = re.compile(r"^\d*<<<?-?$")


_SHELL_ASSIGN_TOKEN_RE = re.compile(r"^[A-Za-z_]\w*=")


def _terminal_path_operands(tokens, text = None) -> "list[tuple[str, bool]]":
    """Absolute file operands a command list touches, as ``(path, writing)``.

    Only commands in the tables above contribute operands, and only tokens that look absolute are
    returned -- a relative operand lands in the session workdir, and a flag value that is not a path
    (``head -n 5``) cannot look absolute. Redirection targets are included whichever command owns
    them, since ``> /abs/file`` truncates that file regardless.
    """
    # Every absolute spelling needs one of these characters; most commands exit here.
    if not any(_PATH_HINT_RE.search(t) for t in tokens):
        return []
    # shlex does not split on < and >, so `echo X>/media/x` is one token.
    tokens = _split_attached_redirections(tokens, text)
    operands: "list[tuple[str, bool]]" = []
    segment: "list[str]" = []
    # Redirections come from the raw text when present, since quoting is what is being decided.
    if text and ("<" in text or ">" in text):
        operands.extend(_raw_redirection_targets(text))

    def flush() -> None:
        if segment:
            operands.extend(_segment_path_operands(segment))
            segment.clear()

    pending_redirect = None
    pending_heredoc = False
    for token in tokens:
        if pending_heredoc:
            # The word after `<<`/`<<<` is not opened, but a substitution inside it still runs.
            pending_heredoc = False
            if not _looks_separator_for_paths(token):
                operands.extend((path, False) for path in _substitution_operand_paths(token))
                continue
        if pending_redirect is not None:
            # `>| /abs` lexes as `>` then `|`; keep waiting for the target.
            if _looks_separator_for_paths(token):
                continue
            writing, pending_redirect = pending_redirect, None
            if _looks_absolute(token):
                operands.append((token, writing))
            continue
        if _looks_separator_for_paths(token):
            flush()
            continue
        if _REDIR_HEREDOC_RE.match(token):
            pending_heredoc = True
            continue
        if _REDIR_WRITE_RE.match(token) or _REDIR_READ_RE.match(token):
            pending_redirect = bool(_REDIR_WRITE_RE.match(token))
            continue
        prefix = _REDIR_PREFIX_RE.match(token)
        if prefix:
            if _REDIR_HEREDOC_RE.match(prefix.group(0)):
                continue
            target = token[prefix.end() :]
            if target and _looks_absolute(target):
                operands.append((target, ">" in prefix.group(0)))
            continue
        segment.append(token)
    flush()
    # Forwarding commands build argv from the pipeline, so every absolute token is a candidate.
    # Substring screen first to skip the per-token lookup.
    if any("xargs" in t or "parallel" in t for t in tokens) and any(
        _token_command_base(t) in _PATH_FORWARDING_COMMANDS for t in tokens
    ):
        # The wrapped command decides the mode: `xargs touch` writes.
        forwarded_write = _forwarded_command_writes(tokens)
        operands.extend((t, forwarded_write) for t in tokens if _looks_absolute(t))
    # Backtick substitutions are scanned in addition to the outer command. A `cd` to an absolute dir
    # is charged as a write when anything after it writes.
    operands.extend(_directory_change_write_targets(tokens))
    nested = _split_backticks(tokens, text)
    if nested != list(tokens):
        operands.extend(_terminal_path_operands(nested, text))
    # Process substitution bodies run as their own command; the lexer splits them across tokens.
    if text and "(" in text:
        for body in _process_substitution_bodies(text):
            try:
                inner = shlex.split(body)
            except ValueError:
                inner = body.split()
            if inner:
                operands.extend(_terminal_path_operands(inner, body))
    return operands


def _process_substitution_bodies(text: str) -> "list[str]":
    """The body of each `<( ... )` / `>( ... )`, matched on BALANCED parentheses.

    A body can hold a substitution of its own (`<(cat $(echo /media/x))`), and a pattern that
    excluded parentheses rejected the whole body when it did. Depth-counted rather than recursive,
    and each body is shorter than the text it came from, so the caller's recursion still terminates.
    """
    bodies: "list[str]" = []
    index = 0
    while index < len(text) - 1:
        if text[index] in "<>" and text[index + 1] == "(":
            depth = 0
            for end in range(index + 1, len(text)):
                if text[end] == "(":
                    depth += 1
                elif text[end] == ")":
                    depth -= 1
                    if depth == 0:
                        bodies.append(text[index + 2 : end])
                        index = end
                        break
            else:
                break
        index += 1
    return bodies


_ATTACHED_REDIR_RE = re.compile(r"(\d*(?:>>|>\||&>>|&>|>|<<<|<<|<))")


def _raw_redirection_targets(text: str) -> "list[tuple[str, bool]]":
    """`(target, writing)` for each redirection whose OPERATOR is unquoted in *text*.

    Read from the raw command rather than the lexed tokens: any rule based on the token alone gets
    either the quoted-data case or the attached-target case wrong. A here-document or here-string
    takes a delimiter or literal data, never a file, so those operators are skipped.
    """
    out: "list[tuple[str, bool]]" = []
    quote = ""
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and quote != "'":
            index += 2
            continue
        if quote:
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in "'\"":
            quote = char
            index += 1
            continue
        if char not in "<>":
            index += 1
            continue
        start = index
        while index < len(text) and text[index] in "<>":
            index += 1
        operator = text[start:index]
        if operator.startswith("<<"):
            continue
        writing = ">" in operator
        while index < len(text) and text[index] in " \t":
            index += 1
        word = ""
        while index < len(text) and (quote or text[index] not in " \t\n;&|()"):
            char = text[index]
            if char == "\\" and quote != "'" and index + 1 < len(text):
                word += text[index + 1]
                index += 2
                continue
            if quote:
                if char == quote:
                    quote = ""
                else:
                    word += char
            elif char in "'\"":
                quote = char
            else:
                word += char
            index += 1
        if word and _looks_absolute(word):
            out.append((word, writing))
    return out


def _token_is_always_quoted(
    token: str,
    text: "str | None",
    *,
    double: bool = True,
) -> bool:
    """Whether EVERY occurrence of *token* in *text* is a quoted one.

    Counted rather than searched: the same word can appear twice, once as data and once as syntax.
    `echo 'x>/media/x'; echo x>/media/x` lexes to that token twice, and a membership test found the
    quoted spelling and left the live redirection of the second command unsplit.
    """
    if not text or not token:
        return False
    occurrences = text.count(token)
    if not occurrences:
        return False
    quoted = text.count(f"'{token}'")
    if double:
        quoted += text.count(f'"{token}"')
    return occurrences == quoted


def _split_backticks(tokens, text = None) -> "list[str]":
    """Break a token carrying a backtick substitution into its own command.

    `` echo `cat /media/x` `` runs that read as a command of its own, but the outer lexer keeps the
    backticks inside ordinary tokens, so the path arrived as an argument of `echo` and was dropped
    with it. `$( ... )` needs none of this: its brackets are punctuation to the lexer already.

    The backtick becomes a separator, which is exactly what it is here -- everything between a pair
    is a command line in its own right, and the segment machinery classifies it as one.
    """
    if not any("`" in token for token in tokens):
        return list(tokens)
    out: "list[str]" = []
    for token in tokens:
        # Single quotes only: substitutions still run inside double quotes.
        if "`" not in token or _token_is_always_quoted(token, text, double = False):
            out.append(token)
            continue
        for index, piece in enumerate(token.split("`")):
            if index:
                out.append("(")
            # Inside double quotes the substitution is one token, so split it into words.
            out.extend(piece.split())
    return out


def _split_attached_redirections(tokens, text = None) -> "list[str]":
    """Break a token that carries a redirection operator inside it into its parts.

    ``echo CHANGED>/media/x`` lexes as one token because ``shlex`` is not given ``<``/``>`` as
    punctuation, which hid the redirection target from the operand scan.

    A QUOTED word is data, not syntax: `printf 'see >/media/private/report'` opens no file, and the
    lexer has already dropped the quotes by the time the token arrives here. Two signs that it was
    quoted are enough to leave it alone -- whitespace inside it, which no redirection operator has,
    and the quoted spelling appearing in the command text the tokens came from.
    """
    if not any(("<" in t or ">" in t) for t in tokens):
        return list(tokens)
    out: "list[str]" = []
    for token in tokens:
        if "<" not in token and ">" not in token:
            out.append(token)
            continue
        if any(ch.isspace() for ch in token) or _token_is_always_quoted(token, text):
            out.append(token)
            continue
        prefix = _REDIR_PREFIX_RE.match(token)
        tail = token[prefix.end() :] if prefix else token
        if prefix and "<" not in tail and ">" not in tail:
            out.append(token)
            continue
        # Several redirections in one token (`</dev/null>/media/x`) must be split.
        parts = [piece for piece in _ATTACHED_REDIR_RE.split(token) if piece]
        out.extend(parts if len(parts) > 1 else [token])
    return out


# Control words start a new command; otherwise the segment is in no table and goes unscanned.
_SHELL_CONTROL_WORDS = frozenset(
    """
    if then elif else fi for while until select do done case esac in { } !
    """.split()
)


def _looks_separator_for_paths(token: str) -> bool:
    return bool(token) and (all(ch in ";&|()" for ch in token) or token in _SHELL_CONTROL_WORDS)


def _segment_path_operands(segment) -> "list[tuple[str, bool]]":
    """Operands of one simple command (no separators, redirections already removed)."""
    index = 0
    # Skip leading assignments and wrappers with their own options/values (`timeout 5`, `env -u V`).
    while index < len(segment):
        token = segment[index]
        base = os.path.basename(token).lower()
        if _SHELL_ASSIGN_TOKEN_RE.match(token):
            index += 1
            continue
        if base in _tools._MULTICALL_BINARIES:
            # `busybox cat` dispatches to the applet named first.
            index += 1
            continue
        if base not in _AUTO_SAFE_WRAPPERS:
            break
        index += 1
        while index < len(segment):
            candidate = segment[index]
            # A lone `-` is env's own option (implies -i); the command still follows.
            if candidate == "-" and base == "env":
                index += 1
                continue
            if candidate.startswith("-") and candidate != "-":
                index += 1
                takes_value = _WRAPPER_VALUE_FLAGS_BY_CMD.get(base)
                if (
                    candidate in (takes_value if takes_value else _WRAPPER_VALUE_FLAGS)
                    and index < len(segment)
                    and not segment[index].startswith("-")
                ):
                    index += 1
                continue
            if base in _WRAPPER_LEADING_VALUE_COMMANDS and _WRAPPER_LEADING_DURATION_RE.match(
                candidate
            ):
                index += 1
                continue
            break
    if index >= len(segment):
        return []
    command = os.path.basename(segment[index]).lower()
    if command.endswith(".exe"):
        command = command[: -len(".exe")]
    args = segment[index + 1 :]
    # Running a binary by path reads it, so report the command word as a read. A trailing
    # separator names a directory, which keeps a sed fragment like `/x/` out.
    launched = (
        [(segment[index], False)]
        if _looks_absolute(segment[index]) and not segment[index].endswith(("/", "\\"))
        else []
    )
    if command in _TEST_COMMANDS:
        operands = list(launched)
        for arg in _test_command_operands(args):
            if _looks_absolute(arg):
                operands.append((arg, False))
            else:
                operands.extend((path, False) for path in _substitution_operand_paths(arg))
        return operands
    # sqlite3 creates/modifies its database unless invoked with -readonly.
    sqlite_write = command == "sqlite3" and not any(
        arg in ("-readonly", "--readonly") for arg in args
    )
    read_cmd = (
        command in _PATH_READ_COMMANDS or command in _PATH_SCRIPT_COMMANDS or command == "sqlite3"
    )
    flag_only = command in _PATH_FLAG_ONLY_COMMANDS
    write_cmd = command in _PATH_WRITE_COMMANDS or sqlite_write
    if (
        write_cmd
        and command in _STDOUT_COMPRESSORS
        and any(
            arg in _STDOUT_FLAGS
            or (arg.startswith("-") and not arg.startswith("--") and "c" in arg.lstrip("-"))
            for arg in args
        )
    ):
        write_cmd = False
        read_cmd = True
    dest_last = command in _PATH_DEST_LAST_COMMANDS
    if not (read_cmd or write_cmd or dest_last or flag_only):
        return launched
    inplace = command in ("sed", "perl") and any(
        arg in _SED_INPLACE_FLAGS or arg.startswith("--in-place") or _clusters_inplace(arg)
        for arg in args
    )
    spec = _PATH_FLAG_SPECS.get(command, {})
    # tar: scan every short-flag token for c/r/u (mode may come later), but only args[0] and
    # single-dash tokens, since filenames can contain those letters.
    if command == "git" and "init" in args:
        # `git init <dir>` creates the repository there.
        after = args[args.index("init") + 1 :]
        target = next((arg for arg in after if not arg.startswith("-")), None)
        return launched + ([(target, True)] if target else [])
    if command in ("7z", "7za", "7zr"):
        creating = bool(args) and args[0].lower() in _SEVENZIP_WRITE_COMMANDS
        archive_from_flag = False
        return launched + _seven_zip_operands(args, creating)
    creating = command in _PATH_ARCHIVE_COMMANDS and (
        any(
            any(mode in arg.lstrip("-") for mode in _ARCHIVE_WRITE_SHORT_MODES)
            for index, arg in enumerate(args)
            if not arg.startswith("--") and (index == 0 or arg.startswith("-"))
        )
        or any(arg.split("=", 1)[0] in _ARCHIVE_CREATE_LONG_FLAGS for arg in args)
    )
    archive_from_flag = command in _PATH_ARCHIVE_COMMANDS and any(
        arg.split("=", 1)[0] in ("-f", "--file")
        or (arg.startswith("-") and not arg.startswith("--") and "f" in arg.lstrip("-"))
        for arg in args
    )
    directory_mode = command == "install" and any(
        arg == "-d"
        or arg == "--directory"
        or (arg.startswith("-") and not arg.startswith("--") and "d" in arg.lstrip("-"))
        for arg in args
    )
    skip = _PATH_ARG_SKIP.get(command, 0)
    operands: "list[tuple[str, bool]]" = list(launched)
    positionals: "list[str]" = []
    pattern_flags = _PATTERN_SUPPLYING_FLAGS.get(command, frozenset())
    pending_flag = None
    for index, arg in enumerate(args):
        if pending_flag is not None:
            flag, pending_flag = pending_flag, None
            _add_flag_operand(operands, spec.get(flag), arg, write_cmd, creating)
            continue
        if arg.startswith("-") and arg != "-":
            # A flag value can be a path or data that merely looks like one (`cut -d /`).
            name, sep, attached = arg.partition("=")
            # The pattern came from a flag, so the first positional is a real operand.
            if name in pattern_flags:
                skip = 0
            if name not in spec:
                name = _resolved_long_flag(spec, name) or name
            kind = spec.get(name)
            if sep and kind:
                _add_flag_operand(operands, kind, attached, write_cmd, creating)
                continue
            if not sep and kind:
                pending_flag = name
                continue
            flag, value = _short_flag_with_value(arg, spec)
            if flag in pattern_flags:
                skip = 0
            if flag and value:
                _add_flag_operand(operands, spec.get(flag), value, write_cmd, creating)
            elif flag:
                pending_flag = flag
            continue
        if skip > 0:
            # For legacy-option-word commands the skip only applies to the first argument.
            if command not in _PATH_SKIP_FIRST_ARG_ONLY or index == 0:
                skip -= 1
                continue
            skip = 0
        positionals.append(arg)
    for position, arg in enumerate(positionals):
        if "=" in arg and not _looks_absolute(arg):
            arg = arg.split("=", 1)[1]
        if dest_last:
            writing = (
                directory_mode
                or command in _PATH_SOURCE_MUTATING_COMMANDS
                or (position == len(positionals) - 1 and len(positionals) > 1)
            )
        elif command in _PATH_DEST_FIRST_COMMANDS:
            writing = position == 0 or any(
                arg == "-m"
                or arg == "--move"
                or (arg.startswith("-") and not arg.startswith("--") and "m" in arg.lstrip("-"))
                for arg in args
            )
        else:
            # Legacy `tar cf out.tar src` puts the archive first; with `-f` the first positional is a source.
            writing = write_cmd or inplace or (creating and position == 0 and not archive_from_flag)
        if not _looks_absolute(arg):
            # The shell expands a substitution here into the operand actually opened.
            operands.extend((path, writing) for path in _substitution_operand_paths(arg))
            continue
        operands.append((arg, writing))
    return operands


# Relative writes after `chdir` land in the new directory; mirrors the terminal `cd` rule.
def _ctor_opens_for_write(node) -> bool:
    """`h5py.File(path, "w")` rewrites the file it names; the default `"r"` mode does not."""
    mode = next(
        (kw.value for kw in _call_keywords(node) if kw.arg == "mode"),
        node.args[1] if len(node.args) > 1 else None,
    )
    return (
        isinstance(mode, ast.Constant)
        and isinstance(mode.value, str)
        and any(char in mode.value for char in "wax+")
    )


def _subprocess_child_writes(tree) -> bool:
    """`subprocess.run(["touch", "f"])` writes, though no python writer call appears in the tree."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (getattr(node.func, "attr", None) or getattr(node.func, "id", None)) not in (
            _PY_PATH_SUBPROCESS_CALLS
        ):
            continue
        head = node.args[0] if node.args else None
        words = getattr(head, "elts", [head])
        first = next(
            (w.value for w in words if isinstance(w, ast.Constant) and isinstance(w.value, str)), ""
        )
        if _command_base_writes(
            _token_command_base(_shell_words(first)[0] if first.split() else "")
        ):
            return True
    return False


def _python_directory_change_targets(tree, operands) -> "list[tuple[str, bool]]":
    """The absolute destination of an `os.chdir` in a snippet that also writes."""
    # `open` is not listed: the operand pass already reports its mode.
    writers = _PY_PATH_WRITE_CALLS | _PY_PATH_CONTENT_FIRST_CALLS
    if (
        not any(writing for _path, writing in operands)
        and not any(
            isinstance(node, ast.Call)
            and (getattr(node.func, "attr", None) or getattr(node.func, "id", None)) in writers
            for node in ast.walk(tree)
        )
        # A child given a relative path leaves no path entry, so its write is only visible here.
        and not _subprocess_child_writes(tree)
    ):
        return []
    targets: "list[tuple[str, bool]]" = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (getattr(node.func, "attr", None) or getattr(node.func, "id", None)) != "chdir":
            continue
        first = node.args[0] if node.args else None
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            targets.append((first.value, True))
    return targets


def _inline_code_operands(source: str) -> "list[tuple[str, bool]]":
    """Operands of a literal `-c` payload, read as python. Anything unparseable yields nothing."""
    try:
        return _python_path_operands(ast.parse(source))
    except Exception:  # noqa: BLE001 - not python, or not parseable: the token scan still applies
        return []


def _shell_words(text: str) -> "list[str]":
    """Split a `shell = True` command line the way the shell does.

    A plain `str.split()` tore `cat '/media/x/My Documents/private.txt'` into two tokens, neither of
    which looked absolute, so the read was never seen. Quotes are the thing being decided here.
    """
    if not text:
        return []
    try:
        words = shlex.split(text)
    except ValueError:
        words = text.split()
    return words or [text]


def _seven_zip_operands(args, creating: bool) -> "list[tuple[str, bool]]":
    """Operands of a 7-Zip invocation: `7z <command> <archive> [files...]`.

    The archive is written whenever the command word adds to, updates, deletes from or renames in
    it, and read otherwise (`x`, `l`, `t`). The remaining positionals are the files it packs.
    """
    positionals = [arg for arg in args[1:] if not arg.startswith("-")]
    operands: "list[tuple[str, bool]]" = []
    for index, arg in enumerate(positionals):
        if _looks_absolute(arg):
            operands.append((arg, creating if index == 0 else False))
        else:
            operands.extend(
                (path, creating if index == 0 else False)
                for path in _substitution_operand_paths(arg)
            )
    return operands


_DIRECTORY_CHANGE_COMMANDS = frozenset({"cd", "pushd"})


def _resolved_long_flag(spec, name: str) -> "str | None":
    """GNU long options accept unambiguous abbreviations, so `cp --targ=` IS `--target-directory=`."""
    if not name.startswith("--") or len(name) < 3:
        return None
    matches = [flag for flag in spec if flag.startswith(name)]
    return matches[0] if len(matches) == 1 else None


def _command_base_writes(base: str) -> bool:
    """Whether a bare command word modifies what it is handed."""
    return (
        base in _PATH_WRITE_COMMANDS
        or base in _PATH_DEST_LAST_COMMANDS
        or base in _PATH_ARCHIVE_COMMANDS
        or base in _EXTRA_WRITE_COMMANDS
    )


_EXTRA_WRITE_COMMANDS = frozenset("rm rmdir touch truncate dd shred sed perl".split())
_FORWARDING_VALUE_FLAGS = frozenset("-a -d -E -e -I -i -L -l -n -P -s".split())


def _forwarded_command_writes(tokens) -> bool:
    """`... | xargs touch` writes every path it is handed; `... | xargs cat` only reads them."""
    for index, token in enumerate(tokens):
        if _token_command_base(token) not in _PATH_FORWARDING_COMMANDS:
            continue
        skip = False
        for candidate in tokens[index + 1 :]:
            if skip:
                skip = False
            elif candidate.startswith("-"):
                skip = candidate in _FORWARDING_VALUE_FLAGS
            else:
                return _command_base_writes(_token_command_base(candidate))
    return False


_INLINE_CODE_COMMANDS = frozenset("python python2 python3 sh bash zsh dash ksh".split())


def _nested_payload_writes(tokens, index: int) -> bool:
    """Whether an interpreter's `-c` payload writes, so the directory it runs in is a write target.

    `cd /models && python -c "open('weights.gguf', 'w')"` writes through a RELATIVE path, which the
    payload scan cannot place and the command-name test never sees.
    """
    payload = next((tokens[i + 1] for i in range(index, len(tokens) - 1) if tokens[i] == "-c"), "")
    if not payload:
        return False
    words = _shell_words(payload)
    if words and _command_base_writes(_token_command_base(words[0])):
        return True
    operands = _inline_code_operands(payload) or _terminal_path_operands(words, payload)
    return any(writing for _path, writing in operands)


def _directory_change_write_targets(tokens) -> "list[tuple[str, bool]]":
    """The absolute destination of a `cd` that is followed by a write, as a write operand."""
    targets: "list[str]" = []
    writes_after = False
    for index, token in enumerate(tokens):
        base = _token_command_base(token)
        if base in _DIRECTORY_CHANGE_COMMANDS:
            operand = tokens[index + 1] if index + 1 < len(tokens) else ""
            if _looks_absolute(operand):
                targets.append(operand)
            continue
        if not targets:
            continue
        if (
            _command_base_writes(base)
            or (base in _INLINE_CODE_COMMANDS and _nested_payload_writes(tokens, index))
            or _REDIR_WRITE_RE.match(token)
            or _REDIR_PREFIX_RE.match(token)
            and ">" in token
        ):
            writes_after = True
    return [(target, True) for target in targets] if writes_after else []


def _serializes_to_second_arg(func, module_aliases: "dict | None" = None) -> bool:
    """True for ``torch.save(obj, path)``-style calls, where the path is the SECOND argument.

    The receiver is resolved through the import aliases first: `import torch as t` makes it `t`,
    which is in no table, and the call would otherwise fall through to the generic writer branch
    that never looks at the second argument.
    """
    if not isinstance(func, ast.Attribute):
        return False
    receiver = func.value
    while isinstance(receiver, ast.Attribute):
        receiver = receiver.value
    if not isinstance(receiver, ast.Name):
        return False
    name = (module_aliases or {}).get(receiver.id, receiver.id)
    return name in _PY_SERIALIZE_SECOND_ARG_MODULES


def _short_flag_with_value(arg: str, spec) -> "tuple[str | None, str | None]":
    """Resolve a short-flag token against a command's flag table.

    Handles the attached form (`sort -o/abs/out`) and the cluster (`tar -cf out.tar`, where only
    the LAST letter takes the value). Returns ``(flag, value)``; ``value`` is None when the value
    is the next token.
    """
    if arg.startswith("--") or len(arg) < 2:
        return None, None
    head = arg[:2]
    if head in spec:
        return head, (arg[2:] or None)
    body = arg[1:]
    if body.isalpha():
        last = "-" + body[-1]
        return (last, None) if last in spec else (None, None)
    # A value can be attached to the cluster (`tar -cf/media/x/out.tar`); split at the first non-letter.
    for index, char in enumerate(body):
        if not char.isalpha():
            break
    else:
        return None, None
    if index < 1:
        return None, None
    last = "-" + body[index - 1]
    value = body[index:]
    return (last, value or None) if last in spec else (None, None)


# shlex keeps a substitution as one non-absolute token, hiding the path the shell passes.
_SUBSTITUTION_RE = re.compile(r"\$\((.*?)\)|`([^`]*)`|\$\{([^{}]*)\}", re.DOTALL)


def _substitution_operand_paths(token: str) -> "list[str]":
    """Absolute paths written literally INSIDE a substitution in `token`.

    Only literals are recovered, which is the common shape (`$(printf /abs)`, `$(echo /abs)`,
    `` `cat /abs` ``). A substitution whose output is genuinely dynamic yields nothing here; that
    remains the documented static-analysis gap rather than something this pretends to solve.
    """
    paths: "list[str]" = []
    for match in _SUBSTITUTION_RE.finditer(token):
        inner = next((group for group in match.groups() if group), "")
        if not inner:
            continue
        try:
            words = shlex.split(inner)
        except ValueError:
            words = inner.split()
        paths.extend(word for word in words if _looks_absolute(word))
    return paths


def _add_flag_operand(operands, kind, value: str, write_cmd: bool, creating: bool) -> None:
    """Record a path supplied as a flag value, with the access that flag implies.

    ``kind`` of "skip" means the value was data, not a path, and is simply consumed.
    """
    if not kind or kind == "skip" or not value:
        return
    if not _looks_absolute(value):
        writing_sub = kind == "write" or (write_cmd and kind != "archive")
        operands.extend((path, writing_sub) for path in _substitution_operand_paths(value))
        return
    if kind == "archive":
        writing = creating
    elif kind == "extract_dir":
        # tar -C is inverse to the archive: extract writes under it, create reads from it.
        writing = not creating
    else:
        writing = kind == "write" or write_cmd
    operands.append((value, writing))


def _terminal_reaches_outside_sandbox(tokens, text: "str | None" = None) -> bool:
    """True when a command list reads or writes an absolute path outside the silent roots.

    *text*, when given, is the command the tokens came from. A POSIX lexer treats a backslash as an
    escape, so `cat C:\\Users\\alice\\notes.txt` arrives here as `C:Usersalicenotes.txt` and every
    Windows absolute path was invisible to the gate. The raw text is re-lexed without that rule when
    it carries a drive or a UNC share, and the operands from both passes are weighed.
    """
    if not any(_ABSOLUTE_HINT_RE.search(token) for token in tokens):
        # A POSIX lex may have eaten the separators, so recheck Windows spellings in the raw text.
        if not (text and _WINDOWS_SPELLING_RE.search(text)):
            return False
    if any(
        _path_needs_approval(path, writing = writing)
        for path, writing in _terminal_path_operands(tokens, text)
    ):
        return True
    if not text or not _WINDOWS_SPELLING_RE.search(text):
        return False
    raw = _lex_keeping_backslashes(text)
    if raw is None or raw == list(tokens):
        return False
    return any(
        _path_needs_approval(path, writing = writing)
        for path, writing in _terminal_path_operands(raw, text)
    )


# Drive-qualified, UNC, or root-relative Windows paths, which a POSIX lexer destroys.
_WINDOWS_SPELLING_RE = re.compile(r"(?:^|[\s'\"=])[A-Za-z]:(?![:\s])|\\\\[^\\/]|(?:^|\s)\\[^\\/\s]")


def _lex_keeping_backslashes(text: str) -> "list[str] | None":
    """Split *text* with the backslash left alone, so a Windows path survives. None if unparseable."""
    try:
        lexer = shlex.shlex(text, posix = False, punctuation_chars = ";&|()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    return [t[1:-1] if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'" else t for t in tokens]


_ABSOLUTE_HINT_RE = re.compile(r"[/~\\]|^[A-Za-z]:")


# Python callables whose first argument (or receiver) is a file read. Ambiguous names are keyed
# on the module: `numpy.load` takes a filename, other serializers' `load` takes a file object.
_PY_QUALIFIED_READ_CALLS = {
    "numpy": frozenset({"load"}),
    "np": frozenset({"load"}),
}
# Readers reached via an instance; the constructor identifies them (`ConfigParser().read(p)`).
_PY_MODULE_OPEN_CTORS = {"FileIO"}


# Constructors that open their first argument; the mode sits where `open`'s does.
_PY_PATH_OPENING_CTORS = frozenset({"ExcelFile", "HDFStore", "File", "Dataset"})


_PY_INSTANCE_READ_CTORS = {
    "ConfigParser": frozenset({"read"}),
    "RawConfigParser": frozenset({"read"}),
    "SafeConfigParser": frozenset({"read"}),
}


# Descending readers; a pattern rooted at `/` reaches the whole host.
_PY_RECURSIVE_READ_CALLS = frozenset("walk rglob glob iglob".split())


_PY_PATH_READ_CALLS = frozenset(
    """
    open_code read_text read_bytes getline getlines loadtxt genfromtxt fromfile read_csv
    read_table read_fwf read_parquet read_json read_excel read_pickle read_feather read_hdf
    read_stata read_sas read_orc read_xml read_html read_sql_table imread connect stat lstat
    exists lexists isfile isdir islink ismount is_file is_dir is_symlink getsize getmtime
    getctime getatime listdir scandir walk iterdir glob iglob rglob samefile realpath readlink
    load_workbook from_file imageio parse iterparse chdir open_memmap memmap get_data read_image
    loadmat
    """.split()
)


_PY_PATH_WRITE_CALLS = frozenset(
    {
        # Source directories are read through `_PY_PATH_KWARGS_BY_CALL`.
        "make_archive",
        "write_text",
        "write_bytes",
        "touch",
        "mkdir",
        "makedirs",
        "rename",
        "renames",
        "replace",
        "remove",
        "unlink",
        "rmdir",
        "removedirs",
        "rmtree",
        "chmod",
        "chown",
        "symlink",
        "symlink_to",
        "hardlink_to",
        "link",
        "truncate",
        "mkfifo",
        "mknod",
        "utime",
    }
)  # extended with `tools._AUTO_UNSAFE_PY_WRITE_METHODS` in `_bind`


# Mode is the second argument, like open(). Extended with `tools._ARCHIVE_CTOR_NAMES` in `_bind`.
_PY_PATH_ARCHIVE_CTORS = frozenset({"TarFile", "tarfile"})


# Destination is the second argument for these modules; numpy.save takes the path first.
_PY_PATH_SERIALIZE_CALLS = frozenset({"save", "dump", "save_file", "save_model"})


_PY_SERIALIZE_SECOND_ARG_MODULES = frozenset(
    {
        "torch",
        "joblib",
        "pickle",
        "cloudpickle",
        "dill",
        "safetensors",
        "st",
        "toml",
        "yaml",
        "json",
    }
)


_PY_PATH_CONTENT_FIRST_CALLS = frozenset({"write_text", "write_bytes", "write"})


# `tempfile` creators write into `dir` when given, else TMPDIR.
_PY_TEMPFILE_CALLS = frozenset(
    """mkstemp mkdtemp NamedTemporaryFile TemporaryFile TemporaryDirectory SpooledTemporaryFile""".split()
)


_PY_PATH_DEST_KWARGS = frozenset(
    {
        "dst",
        "dest",
        "destination",
        "target",
        "output",
        "out",
        "save_directory",
        "f",
        "extract_dir",
    }
)


_PY_PATH_SUBPROCESS_CALLS = frozenset(
    {
        "run",
        "Popen",
        "call",
        "check_call",
        "check_output",
        "getoutput",
        "getstatusoutput",
        "system",
        "popen",
    }
)


# Destination-second callables that also remove their source.
_PY_PATH_MOVE_CALLS = frozenset({"move", "rename", "renames", "replace"})


# `os.symlink` target is checked as a write: the new name writes to it.
_PY_PATH_LINK_CALLS = frozenset({"link", "symlink"})


_PY_PATH_EXTRACT_CALLS = frozenset({"extract", "extractall"})


_PY_PATH_DEST_SECOND_CALLS = frozenset(
    {
        "copy",
        "copy2",
        "copyfile",
        "copytree",
        "copymode",
        "copystat",
        "move",
        "rename",
        "renames",
        "replace",
        "link",
        "symlink",
        "unpack_archive",
    }
)


_PY_PATH_KWARGS = (
    "path",
    "file",
    "filename",
    "filepath",
    "file_path",
    "fname",
    "name",
    "src",
    "source",
    "dst",
    "dest",
    "destination",
    "output",
    "out",
    "path_or_buf",
    "filepath_or_buffer",
    "save_directory",
    "directory",
    "folder",
    "dirname",
    "top",
)


# Path parameters only these callables use; elsewhere the same keyword may not be a path.
_PY_PATH_KWARGS_BY_CALL = {
    "read": ("filenames",),
    "open": ("fp",),
    "imread": ("fname",),
    "load": ("file",),
    "glob": ("pathname", "root_dir"),
    "iglob": ("pathname", "root_dir"),
    "listdir": ("path",),
    "scandir": ("path",),
    "walk": ("top",),
    "read_excel": ("io",),
    "make_archive": ("root_dir", "base_dir"),
}


# Extra positional source paths, e.g. make_archive's root_dir/base_dir.
_PY_PATH_EXTRA_READ_POSITIONS = {"make_archive": (2, 3)}


# Built from the tables so a name added to one is aliasable at once.
_PY_ALIASABLE_PATH_CALLS = (
    frozenset({"open", "fdopen"})
    | _PY_PATH_READ_CALLS
    | _PY_PATH_WRITE_CALLS
    | _PY_PATH_DEST_SECOND_CALLS
    | _PY_PATH_SERIALIZE_CALLS
    | _PY_PATH_CONTENT_FIRST_CALLS
    | _PY_PATH_ARCHIVE_CTORS
    | _PY_PATH_SUBPROCESS_CALLS
)


def _python_function_aliases(tree, module_aliases: "dict | None" = None) -> dict:
    """Local name -> real function, for `from io import open as fopen` and `reader = open`.

    Every MODELED path call is covered, not only the open-like ones: `from pandas import read_csv as
    rc` leaves a plain `Name` under a name in no table, so the call is never dispatched and its path
    argument is never looked at. Restricting this to `open` left every other reader and writer in
    the tables reachable under an alias.

    A plain assignment binds the same identity as an import does, so `reader = open` and
    `rc = pandas.read_csv` are followed too. An attribute is only followed when its receiver is a
    modelled module, so `sock.open` and `self.open` stay unmodelled rather than resolving to the
    builtin. A name assigned more than once is dropped: which function it holds at the call is not
    answerable here, and guessing either way would be wrong half the time.
    """
    aliases: dict = {}
    assigned: dict = {}
    seen_twice: "set[str]" = set()
    for node in _tree_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            for entry in node.names:
                if entry.asname and entry.name in _PY_ALIASABLE_PATH_CALLS:
                    aliases[entry.asname] = entry.name
            continue
        if isinstance(node, ast.AnnAssign):
            if node.value is None or not isinstance(node.target, ast.Name):
                continue
            node = ast.Assign(targets = [node.target], value = node.value)
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if isinstance(value, ast.Name):
            real = value.id
        elif isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
            receiver = value.value.id
            receiver = (module_aliases or {}).get(receiver, receiver)
            real = value.attr if receiver in _PY_MODULE_PATH_RECEIVERS else None
        else:
            real = None
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            # A rebinding does not undo earlier calls: keep the first modelled binding (fail closed).
            if target.id in seen_twice:
                if real and real != target.id:
                    assigned.setdefault(target.id, real)
                continue
            seen_twice.add(target.id)
            # Recorded even if not modelled yet; the chain below resolves or drops it.
            if real and real != target.id:
                assigned[target.id] = real
    for name, real in assigned.items():
        aliases.setdefault(name, real)
    # Follow alias chains to the modelled name; a cycle stops when a name repeats.
    resolved: dict = {}
    for name, real in aliases.items():
        seen = {name}
        while real in aliases and real not in seen:
            seen.add(real)
            real = aliases[real]
        if real in _PY_ALIASABLE_PATH_CALLS:
            resolved[name] = real
    return resolved


def _python_module_aliases(tree) -> dict:
    """Local name -> real module, for `import io as stream` and `import os.path as p`.

    Only the ROOT module is recorded, which is what the receiver tables are keyed on.

    `from PIL import Image as I` counts too: `Image` is itself a modelled receiver, so `I.open(p)`
    has to resolve back to it or the call reads as a Path-style method and the filename argument is
    never looked at.
    """
    aliases: dict = {}
    assigned: "list[tuple[str, str]]" = []
    for node in _tree_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            for entry in node.names:
                if entry.asname and entry.name in _PY_MODULE_PATH_RECEIVERS:
                    aliases[entry.asname] = entry.name
            continue
        # `stream = io` aliases the module; resolved after the walk since the import may come later.
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id != node.value.id:
                    assigned.append((target.id, node.value.id))
        if not isinstance(node, ast.Import):
            continue
        for entry in node.names:
            root = entry.name.split(".", 1)[0]
            aliases[entry.asname or root] = root
    # Only copies resolving to a modelled module count; names bound twice are dropped.
    bound_twice = {name for name, _ in assigned if sum(1 for n, _ in assigned if n == name) > 1}
    for name, source in assigned:
        if name in bound_twice or name in aliases:
            continue
        seen = {name}
        while source in dict(assigned) and source not in seen and source not in aliases:
            seen.add(source)
            source = dict(assigned)[source]
        root = aliases.get(source, source)
        if root in _PY_MODULE_PATH_RECEIVERS:
            aliases[name] = root
    return aliases


def _chained_instance_reader_methods(receiver, module_aliases: "dict | None" = None) -> frozenset:
    """Reader methods for `ConfigParser().read(p)`, where the receiver is the constructor call."""
    if not isinstance(receiver, ast.Call):
        return frozenset()
    func = receiver.func
    if isinstance(func, ast.Attribute):
        name = func.attr
    elif isinstance(func, ast.Name):
        name = (module_aliases or {}).get(func.id, func.id)
    else:
        return frozenset()
    return _PY_INSTANCE_READ_CTORS.get(name, frozenset())


# `ZipFile.write(src)` / `TarFile.add(src)` read a file from disk, unlike `f.write(data)`.
_PY_ARCHIVE_SOURCE_CALLS = frozenset({"write", "add"})
_PY_ARCHIVE_MEMBER_CTORS = frozenset({"ZipFile", "TarFile"})


def _python_archive_ctor_names(tree) -> "set[str]":
    """Local names imported FROM an archive module that construct one.

    `from tarfile import open as topen` binds a constructor under a name whose resolved spelling is
    `open`, which is the builtin everywhere else, so the module it came from is what identifies it.
    """
    names: "set[str]" = set()
    for node in _tree_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("tarfile", "zipfile"):
            for entry in node.names:
                if entry.name in ("open", "ZipFile", "TarFile"):
                    names.add(entry.asname or entry.name)
    return names


def _is_archive_ctor_call(
    node,
    module_aliases = None,
    local_ctors = frozenset(),
) -> bool:
    """True for `zipfile.ZipFile(...)`, `ZipFile(...)`, `tarfile.open(...)` and their aliases."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        receiver = func.value
        base = receiver.id if isinstance(receiver, ast.Name) else ""
        base = (module_aliases or {}).get(base, base)
        return func.attr in _PY_ARCHIVE_MEMBER_CTORS or (
            func.attr == "open" and base in ("tarfile", "zipfile")
        )
    if isinstance(func, ast.Name):
        return (
            func.id in local_ctors
            or (module_aliases or {}).get(func.id, func.id) in _PY_ARCHIVE_MEMBER_CTORS
        )
    return False


def _python_archive_object_names(
    tree,
    module_aliases = None,
    local_ctors = frozenset(),
) -> "set[str]":
    """Local names holding a member-based archive, bound by assignment or by `with ... as`."""
    names: "set[str]" = set()

    def bind(target, value) -> None:
        if isinstance(target, ast.Name) and _is_archive_ctor_call(
            value, module_aliases, local_ctors
        ):
            names.add(target.id)

    for node in _tree_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bind(node.target, node.value)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                bind(item.optional_vars, item.context_expr)
    return names


def _python_instance_reader_names(tree) -> dict:
    """`local name -> reader methods`, for a reader reached through an instance.

    `cfg = ConfigParser(); cfg.read(p)` opens p, and `read` on any other receiver is an ordinary
    method, so the CONSTRUCTOR is what identifies it. Import aliases count. The chained
    `ConfigParser().read(p)` binds no name and is resolved separately, at the call site.
    """
    ctors = dict(_PY_INSTANCE_READ_CTORS)
    for node in _tree_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            for entry in node.names:
                if entry.asname and entry.name in _PY_INSTANCE_READ_CTORS:
                    ctors[entry.asname] = _PY_INSTANCE_READ_CTORS[entry.name]
    names: dict = {}
    for node in _tree_nodes(tree):
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            if not isinstance(node.target, ast.Name):
                continue
            node = ast.Assign(targets = [node.target], value = node.value)
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        ctor = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        methods = ctors.get(ctor)
        if not methods:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names[target.id] = methods
    return names


def _python_qualified_read_aliases(tree) -> "set[str]":
    """Local names bound from a module whose reader is only modelled QUALIFIED.

    `numpy.load` takes a filename while every other `load` in these tables takes an open file, so the
    name is keyed on its module. `from numpy import load` and `from numpy import load as read_array`
    drop that module, and the call arrives as a bare Name the reader table deliberately omits.
    """
    aliases: "set[str]" = set()
    for node in _tree_nodes(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = (node.module or "").split(".")[0]
        readers = _PY_QUALIFIED_READ_CALLS.get(module)
        if not readers:
            continue
        for entry in node.names:
            if entry.name in readers:
                aliases.add(entry.asname or entry.name)
    return aliases


# Filesystem ops only on a path receiver: `text.replace(a, b)` touches no file.
_PY_AMBIGUOUS_PATH_METHODS = frozenset({"replace", "remove"})


def _receiver_is_a_path_object(receiver, ctors, path_objects) -> bool:
    """True when the receiver of an ambiguous method is a `Path`-like value.

    A direct `Path(p).replace(q)`, a name bound to one, or a chain off either. Anything else -- a
    str, a list, a DataFrame -- is left alone, which is what the pre-existing analyzer did by
    keeping these names qualified.
    """
    while isinstance(receiver, ast.Attribute):
        receiver = receiver.value
    if isinstance(receiver, ast.Call):
        func = receiver.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        return name in ctors
    if isinstance(receiver, ast.Name):
        return receiver.id in path_objects
    return False


def _python_path_object_names(tree, ctors) -> "set[str]":
    """Local names bound to a `Path`-like object, by assignment or by `with ... as`."""
    names: "set[str]" = set()

    def bind(target, value) -> None:
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
            return
        func = value.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in ctors:
            names.add(target.id)

    for node in _tree_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bind(node.target, node.value)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                bind(item.optional_vars, item.context_expr)
    return names


def _python_path_fold_aliases(tree) -> "tuple[set, set]":
    """`(path constructor names, os.path.join names)`, including the local names imports bind them to.

    `from pathlib import Path as P` leaves `P(...)` under a name the fold does not know, so the call
    resolves to nothing and the path it builds is never checked. The main analyzer already collects
    these for its own fold; the operand pass needs the same two sets.
    """
    ctors = set(_PATH_CTORS)
    joins: "set[str]" = set()
    for node in _tree_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for entry in node.names:
                local = entry.asname or entry.name
                if entry.name in _PATH_CTORS:
                    ctors.add(local)
                elif entry.name == "join" and (module == "os.path" or module.endswith(".path")):
                    joins.add(local)
            continue
        # `P = Path` / `j = os.path.join` bind like an import alias.
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if isinstance(value, ast.Name):
            source = value.id
        elif isinstance(value, ast.Attribute):
            source = value.attr
        else:
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id == source:
                continue
            if source in ctors:
                ctors.add(target.id)
            elif source in joins or source == "join":
                joins.add(target.id)
    return ctors, joins


def _python_path_bindings(
    tree,
    ctors = None,
    joins = None,
) -> dict:
    """Names bound to a foldable path (`p = '/media/x'`, `p := Path('/media') / 'x'`), so a read
    through the variable folds to the same path a literal would.

    A name bound more than once keeps EVERY path it was bound to, not the last one: rebinding
    ``p`` after the read must not reclassify the read that already happened.

    EVERY target of a chained assignment is bound, not just the first: `src = backup = '/media/x'`
    has to make `open(src)` foldable, or the read runs unprompted.
    """
    bindings: dict = {}
    extra: "dict[str, list[str]]" = {}
    for node in _tree_nodes(tree):
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, ast.NamedExpr):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        # Tuple targets bind elementwise, per target.
        pairs = []
        for target in targets:
            if isinstance(target, ast.Tuple) and isinstance(value, (ast.Tuple, ast.List)):
                pairs.extend(zip(target.elts, value.elts))
            else:
                pairs.append((target, value))
        for target, bound in pairs:
            if not isinstance(target, ast.Name):
                continue
            try:
                folded = _folded_path(bound, bindings, ctors, joins)
            except Exception:  # noqa: BLE001 - folding is best effort
                continue
            # Substitute alternate bindings of dependencies one at a time (linear, not cartesian).
            derived: "list[str]" = []
            if extra:
                for used in {n.id for n in ast.walk(bound) if isinstance(n, ast.Name)}:
                    for alternate in _capped_alternates(extra.get(used, ())):
                        try:
                            other = _folded_path(bound, {**bindings, used: alternate}, ctors, joins)
                        except Exception:  # noqa: BLE001 - folding is best effort
                            continue
                        if isinstance(other, str) and other and "\x00" not in other:
                            derived.append(other)
            if not isinstance(folded, str) or not folded or "\x00" in folded:
                for candidate in _capped_alternates(derived):
                    extra.setdefault(target.id, []).append(candidate)
                continue
            if target.id in bindings and bindings[target.id] != folded:
                extra.setdefault(target.id, []).append(folded)
            else:
                bindings[target.id] = folded
            for candidate in _capped_alternates(derived):
                if candidate != bindings.get(target.id):
                    extra.setdefault(target.id, []).append(candidate)
    if extra:
        bindings[_REBOUND_PATHS_KEY] = extra
    return bindings


_REBOUND_PATHS_KEY = "\x00__rebound__"


# Bounds alternates when a loop rebinds a name many times.
_MAX_REBOUND_ALTERNATES = 8


_MAX_REBOUND_NAMES = 3


def _capped_alternates(values) -> "list[str]":
    """Bound a list of candidate paths WITHOUT dropping the ones that can need approval.

    A plain slice discards by position, so eight benign reassignments ahead of
    `base = "/media/private"` hid the only value that mattered. Absolute candidates are kept first
    and the cap is spent on the remainder, so the bound limits work rather than coverage.
    """
    values = list(values)
    if len(values) <= _MAX_REBOUND_ALTERNATES:
        return values

    # Ranked by approval cost so the cap never drops a value that needs it. Write-silent roots are
    # a subset of read-silent ones, so the ranking covers both modes.
    def rank(value: str) -> int:
        if _path_needs_approval(value):
            return 0
        return 1 if _path_needs_approval(value, writing = True) else 2

    ranked: "list[list[str]]" = [[], [], []]
    for value in values:
        ranked[rank(value)].append(value)
    return (ranked[0] + ranked[1] + ranked[2])[:_MAX_REBOUND_ALTERNATES]


def _sequence_elements(node, containers) -> "list":
    """The elements of a literal sequence, or the node itself when it is not one.

    A NAME holding one counts: `paths = ["/media/x"]; cfg.read(paths)` passes the same list the
    inline form passes, and handing the bare name to the fold resolved nothing. The paths come from
    the literal containers already collected for subscripts, so there is one place that knows what a
    name holds.
    """
    if node is None:
        return []
    if isinstance(node, (ast.List, ast.Tuple)):
        return list(node.elts)
    if isinstance(node, ast.Name) and node.id in containers:
        return [ast.Constant(value = path) for path in containers[node.id]]
    return [node]


def _sqlite_opens_read_only(node, given = None) -> bool:
    """True when a sqlite connection is provably read-only: a `file:...?mode=ro` URI."""
    first = given if given is not None else (node.args[0] if node.args else None)
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        return False
    lowered = first.value.lower()
    return lowered.startswith("file:") and ("mode=ro" in lowered or "immutable=1" in lowered)


def _call_keywords(node) -> "list":
    """A call's keywords, with a LITERAL `**{...}` splat expanded into the keywords it stands for.

    `pd.read_csv(**{"filepath_or_buffer": "/media/x"})` passes the path under its own parameter
    name; the splat arrives as a keyword whose `arg` is None, so the name was never matched.
    """
    keywords = list(node.keywords)
    for keyword in node.keywords:
        if keyword.arg is not None or not isinstance(keyword.value, ast.Dict):
            continue
        for key, value in zip(keyword.value.keys, keyword.value.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keywords.append(ast.keyword(arg = key.value, value = value))
    return keywords


def _python_fileinput_readers(tree) -> "set[str]":
    """Local names bound by `from fileinput import input` / `FileInput`, with or without an alias.

    Tracked by PROVENANCE rather than by name: a bare `input` is the builtin prompt on its own, and
    only an import from this module makes it the reader that opens a file.
    """
    names: "set[str]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "fileinput":
            for entry in node.names:
                if entry.name in ("input", "FileInput"):
                    names.add(entry.asname or entry.name)
    return names


def _python_literal_containers(tree) -> dict:
    """Name -> the absolute-looking strings a literal list, tuple or dict assigned to it holds."""
    containers: dict = {}
    for node in _tree_nodes(tree):
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        paths = _literal_container_paths(value)
        if not paths:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                containers.setdefault(target.id, []).extend(paths)
    return containers


def _literal_container_paths(node) -> "list[str]":
    """The absolute-looking strings a literal list, tuple or dict holds, values only for a dict."""
    if isinstance(node, (ast.List, ast.Tuple)):
        elements = node.elts
    elif isinstance(node, ast.Dict):
        elements = node.values
    else:
        return []
    return [
        element.value
        for element in elements
        if isinstance(element, ast.Constant)
        and isinstance(element.value, str)
        and _looks_absolute(element.value)
    ]


def _subscript_literal_paths(node, containers) -> "list[str]":
    """The paths a subscript can resolve to, when everything about it is a literal.

    A constant index resolves to the ONE element it names, which is both exact and free of the cap
    below. Otherwise every path the container holds is a candidate, ranked so the ones that need
    approval survive the cap: a container of nine paths whose last is the outside one would
    otherwise have lost it to eight read-silent ones ahead of it.
    """
    target = node.value
    paths = (
        list(containers.get(target.id, ()))
        if isinstance(target, ast.Name)
        else _literal_container_paths(target)
    )
    index = node.slice
    if isinstance(index, ast.Constant) and isinstance(index.value, int) and paths:
        # Resolves exactly only when the container holds nothing but paths.
        exact = _exact_indexed_path(target, containers, index.value)
        if exact is not None:
            return [exact]
    return _capped_alternates(paths)


def _exact_indexed_path(target, containers, index: int) -> "str | None":
    """The element a constant index names, when every element of the container is an absolute path."""
    if isinstance(target, (ast.List, ast.Tuple)):
        elements = target.elts
    else:
        return None
    if not all(
        isinstance(element, ast.Constant) and isinstance(element.value, str) for element in elements
    ):
        return None
    try:
        value = elements[index].value
    except IndexError:
        return None
    return value if _looks_absolute(value) else None


def _python_path_operands(tree) -> "list[tuple[str, bool]]":
    """Absolute path operands a python snippet reads or writes, as ``(path, writing)``.

    Reuses ``_folded_path`` so a path assembled from literals, an f-string, ``os.path.join`` or a
    ``Path`` chain resolves the same way the credential scan resolves it. A path the folder cannot
    resolve yields nothing here; the dynamic-alias checks elsewhere cover those.
    """
    ctors, joins = _python_path_fold_aliases(tree)
    qualified_readers = _python_qualified_read_aliases(tree)
    instance_readers = _python_instance_reader_names(tree)
    bindings = _python_path_bindings(tree, ctors, joins)
    rebound = bindings.get(_REBOUND_PATHS_KEY) or {}
    module_aliases = _python_module_aliases(tree)
    function_aliases = _python_function_aliases(tree, module_aliases)
    # Function aliases (`from zipfile import ZipFile as Z`) count like module ones.
    archive_aliases = {**module_aliases, **function_aliases}
    archive_ctors = _python_archive_ctor_names(tree)
    path_objects = _python_path_object_names(tree, ctors)
    archive_objects = _python_archive_object_names(tree, archive_aliases, archive_ctors)
    containers = _python_literal_containers(tree)
    fileinput_readers = _python_fileinput_readers(tree)
    operands: "list[tuple[str, bool]]" = []

    def add_subprocess_operands(call) -> None:
        """Screen a child process's argv with the terminal operand scanner.

        Handing the words to the shell scan keeps the read/write distinction: `["cp", "a", "/etc/b"]`
        is a write to /etc, not a read of it. `shell = True` passes one command line, which splits
        the same way.
        """
        words: "list[str]" = []
        for argument in (*call.args, *(k.value for k in call.keywords)):
            for piece in ast.walk(argument):
                if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                    words.extend(_shell_words(piece.value))
                elif isinstance(piece, ast.Name):
                    # Split a folded command string so `subprocess.run(cmd, shell = True)` is scanned.
                    folded = bindings.get(piece.id)
                    if isinstance(folded, str) and folded:
                        words.extend(_shell_words(folded))
                    for path in rebound.get(piece.id, ()):
                        words.extend(_shell_words(path))
        # `executable =` launches that binary; argv[0] is only the child's name for itself.
        for keyword in _call_keywords(call):
            if keyword.arg == "executable":
                add(keyword.value, False)
        # argv[0] is the binary that runs, even if a known command name appears later.
        head = call.args[0] if call.args else None
        if isinstance(head, (ast.List, ast.Tuple)) or (
            isinstance(head, ast.Name) and head.id in containers
        ):
            argv0 = next(iter(_sequence_elements(head, containers)), None)
            if argv0 is not None:
                add(argv0, False)
        # `python -c` carries code, scanned by the python scan.
        elements = _sequence_elements(call.args[0], containers) if call.args else []
        literals = [
            e.value for e in elements if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
        for index, literal in enumerate(literals[:-1]):
            if (
                literal in ("-c", "-e")
                and _token_command_base(literals[0]) in _PATH_SCRIPT_COMMANDS
            ):
                operands.extend(_inline_code_operands(literals[index + 1]))
        if words:
            from_command = _terminal_path_operands(words)
            operands.extend(from_command)
            # Only for an unrecognised command: `["echo", "/home/x"]` is as harmless as at a terminal.
            if not from_command and not any(
                _token_command_base(word) in _classified_terminal_commands() for word in words
            ):
                operands.extend((word, False) for word in words if _looks_absolute(word))

    def add(node, writing: bool) -> None:
        if node is None:
            return
        if isinstance(node, ast.NamedExpr):
            node = node.value
        # Only literal sequences are unpacked, so nothing dynamic is guessed.
        if isinstance(node, ast.Starred):
            for element in _sequence_elements(node.value, containers):
                if element is not node.value:
                    add(element, writing)
            return
        try:
            folded = _folded_path(node, bindings, ctors, joins)
        except Exception:  # noqa: BLE001
            return
        if isinstance(folded, str) and folded:
            operands.append((folded, writing))
        elif isinstance(node, ast.Subscript):
            # Every literal in the container counts: the index is not always constant.
            for path in _subscript_literal_paths(node, containers):
                operands.append((path, writing))
        # Order is not tracked, so every value a rebound name held counts.
        if isinstance(node, ast.Name) and node.id in rebound:
            operands.extend((path, writing) for path in rebound[node.id])
            return
        # Substitute one rebound name at a time inside larger expressions.
        names = [
            piece.id
            for piece in ast.walk(node)
            if isinstance(piece, ast.Name) and piece.id in rebound
        ]
        for rebound_name in list(dict.fromkeys(names))[:_MAX_REBOUND_NAMES]:
            for alternate in rebound[rebound_name]:
                try:
                    refolded = _folded_path(
                        node, {**bindings, rebound_name: alternate}, ctors, joins
                    )
                except Exception:  # noqa: BLE001 - folding is best effort
                    continue
                if isinstance(refolded, str) and refolded and refolded != folded:
                    operands.append((refolded, writing))

    for node in _tree_nodes(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = function_aliases.get(func.id, func.id)
        else:
            continue
        first = node.args[0] if node.args else None
        second = node.args[1] if len(node.args) > 1 else None
        is_method = isinstance(func, ast.Attribute)
        # Resolved once: each path-receiver branch below has a module-function counterpart.
        receiver_name = (
            module_aliases.get(getattr(func.value, "id", ""), getattr(func.value, "id", ""))
            if is_method
            else ""
        )
        module_receiver = is_method and receiver_name in _PY_MODULE_PATH_RECEIVERS
        writing_default = name in _PY_PATH_WRITE_CALLS
        if name in _PY_MODULE_OPEN_CTORS and receiver_name in _PY_MODULE_OPEN_RECEIVERS:
            add(first, _open_call_writes(node, mode_index = 1))
        elif name in ("open", "fdopen"):
            # The mode decides read vs write; as a method the mode is the first argument. Module functions
            # (io.open, gzip.open) take the path first, so their receiver is not a path.
            receiver = receiver_name
            # A dotted receiver has no `.id`; the trailing name is what the table is keyed on.
            tail = func.value.attr if is_method and isinstance(func.value, ast.Attribute) else ""
            path_is_receiver = is_method and not (
                receiver in _PY_MODULE_OPEN_RECEIVERS or tail in _PY_MODULE_OPEN_RECEIVERS
            )
            writing = _open_call_writes(node, mode_index = 0 if path_is_receiver else 1) or (
                receiver in ("os", "posix")
            )
            writing_default = writing
            add(func.value if path_is_receiver else first, writing)
        elif name in _PY_PATH_ARCHIVE_CTORS:
            writing = _open_call_writes(node, mode_index = 1)
            writing_default = writing
            add(first, writing)
        elif name in _PY_PATH_SUBPROCESS_CALLS:
            # A child is not bound by this scan, so screen its argv as a command line.
            add_subprocess_operands(node)
        elif (
            name in _PY_AMBIGUOUS_PATH_METHODS
            and is_method
            and not module_receiver
            and not _receiver_is_a_path_object(func.value, ctors, path_objects)
        ):
            # Only a Path-like receiver makes these filesystem calls.
            continue
        elif name in _PY_PATH_DEST_SECOND_CALLS:
            # As a method the receiver is the source; `os.rename(...)` is the function form.
            receiver_is_path = is_method and not module_receiver
            # A move removes its source, so that side is a write too; a copy leaves it alone.
            add(
                func.value if receiver_is_path else first,
                name in _PY_PATH_MOVE_CALLS or name in _PY_PATH_LINK_CALLS,
            )
            add(first if receiver_is_path else second, True)
        elif name in _PY_PATH_SERIALIZE_CALLS and _serializes_to_second_arg(func, module_aliases):
            # torch.save / joblib.dump put the destination second, unlike numpy.save.
            add(second, True)
        elif name in _PY_PATH_EXTRACT_CALLS:
            # Extraction destination is the first arg for extractall, the second for extract.
            add(second if name == "extract" else first, True)
            for keyword in _call_keywords(node):
                if keyword.arg == "path":
                    add(keyword.value, True)
        elif (
            name in _PY_ARCHIVE_SOURCE_CALLS
            and is_method
            and (
                receiver_name in archive_objects
                or _is_archive_ctor_call(func.value, module_aliases)
            )
        ):
            add(first, False)
        elif name in _PY_PATH_CONTENT_FIRST_CALLS:
            # The first argument is data, not a path; only the receiver is.
            add(func.value if is_method else None, True)
        elif name in _PY_PATH_WRITE_CALLS:
            add(first, True)
            if is_method:
                add(func.value, True)
        elif is_method and (
            name in instance_readers.get(receiver_name, ())
            or name in _chained_instance_reader_methods(func.value, module_aliases)
        ):
            for element in _sequence_elements(first, containers):
                add(element, False)
        elif name in _PY_QUALIFIED_READ_CALLS.get(receiver_name, ()) or (
            not is_method and name in qualified_readers
        ):
            # Qualified: `numpy.load` takes a path, `json/pickle/torch.load` take an open file.
            add(first, False)
        elif (name in ("input", "FileInput") and is_method and receiver_name == "fileinput") or (
            not is_method and func.id in fileinput_readers
        ):
            # Qualified so builtin input() is not a file read; `files =` is the keyword form.
            given = first
            if given is None:
                given = next((kw.value for kw in _call_keywords(node) if kw.arg == "files"), None)
            for element in _sequence_elements(given, containers):
                add(element, False)
        elif name in ("connect", "Connection") and receiver_name in ("sqlite3", "apsw"):
            # A connection creates the file if missing, so it is a write unless the URI says otherwise.
            given = first
            if given is None:
                given = next(
                    (kw.value for kw in _call_keywords(node) if kw.arg in ("database", "filename")),
                    None,
                )
            add(given, not _sqlite_opens_read_only(node, given))
        elif name in _PY_TEMPFILE_CALLS:
            # Without `dir` the file lands in the sandbox TMPDIR, which is silent.
            add(next((kw.value for kw in _call_keywords(node) if kw.arg == "dir"), None), True)
        elif name in _PY_PATH_OPENING_CTORS:
            # The mode applies to the keyword spelling of the path too.
            writing_default = _ctor_opens_for_write(node)
            add(first, writing_default)
        elif name in _PY_PATH_READ_CALLS:
            start = len(operands)
            add(first, False)
            if is_method:
                add(func.value, False)
            if name in _PY_RECURSIVE_READ_CALLS:
                # A recursive reader rooted at `/` descends the whole host, unlike `ls /`.
                for path, _writing in operands[start:]:
                    if path.rstrip("/\\") == "":
                        operands.append((path.rstrip("/\\") + "/**", False))
        else:
            continue
        for position in _PY_PATH_EXTRA_READ_POSITIONS.get(name, ()):
            if position < len(node.args):
                add(node.args[position], False)
        for keyword in _call_keywords(node):
            if keyword.arg in _PY_PATH_KWARGS_BY_CALL.get(name, ()):
                for element in _sequence_elements(keyword.value, containers):
                    add(element, False)
            elif keyword.arg in _PY_PATH_DEST_KWARGS:
                add(keyword.value, True)
            elif keyword.arg in _PY_PATH_KWARGS:
                # Keyword `file =` carries the positional form's mode.
                add(keyword.value, writing_default)
    operands.extend(_python_directory_change_targets(tree, operands))
    return operands


def _python_reaches_outside_sandbox(tree, code = None) -> bool:
    """True when python code reads or writes an absolute path outside the silent roots."""
    # No separator, tilde or drive colon means no absolute path; skip the AST walks.
    if isinstance(code, str) and not _ABSOLUTE_HINT_RE.search(code):
        return False
    try:
        operands = _python_path_operands(tree)
    except Exception:  # noqa: BLE001 - an unexpected AST shape must not crash the classifier
        return False
    return any(_path_needs_approval(path, writing = writing) for path, writing in operands)


def _open_call_writes(node, *, mode_index: int) -> bool:
    """Write check for an open-like call whose mode sits at ``mode_index``.

    ``open(file, mode)`` carries it at 1; ``Path(p).open(mode)`` at 0, because the receiver is the
    path. Reading the wrong position silently turns a write into a read.
    """
    if _has_kwarg_splat(node):
        return True
    if any(isinstance(a, ast.Starred) for a in node.args):
        return True
    mode = node.args[mode_index] if len(node.args) > mode_index else None
    for kw in node.keywords or []:
        if kw.arg == "mode":
            mode = kw.value
    return _mode_arg_writes(mode)


# Placed last so `_bind` finds this module complete whichever module is imported first.
from . import tools as _tools  # noqa: E402,F401
