# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read-only, bounded Git status and diff review for persisted project roots."""

from __future__ import annotations

import errno
import hashlib
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from .git_context import AgentWorkspaceError, ProjectWorkspace, project_workspace_access
from .git_files import read_project_file


DIFF_MODES = frozenset({"head", "staged", "unstaged"})
DEFAULT_MAX_BYTES = 512_000
MAX_MAX_BYTES = 2_000_000
MAX_STATUS_BYTES = 1_000_000
MAX_CONFIG_BYTES = 256_000
MAX_FILES = 5_000
MAX_HUNKS = 20_000
MAX_LINES = 200_000
MAX_LINE_CHARS = 64_000
MAX_UNTRACKED_FILE_BYTES = 128_000
MAX_CONFIG_DRIVERS = 1_000
MANIFEST_VERSION = 1

_CONFLICT_CODES = frozenset({b"DD", b"AU", b"UD", b"UA", b"DU", b"AA", b"UU"})
_DRIVER_KEY = re.compile(
    rb"^(?P<kind>filter|diff|merge)\.(?P<driver>[A-Za-z0-9][A-Za-z0-9._-]{0,127})\."
    rb"(?P<option>clean|smudge|process|required|command|textconv|driver)$"
)
_RAW_HEADER = re.compile(
    rb"^:(?P<old_mode>[0-7]{6}) (?P<new_mode>[0-7]{6}) "
    rb"(?P<old_sha>[0-9a-f]+) (?P<new_sha>[0-9a-f]+) "
    rb"(?P<status>[A-Z][0-9]{0,3})$"
)
_HUNK_HEADER = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_lines>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_lines>\d+))? @@(?P<label>.*)$"
)
_OBJECT_ID = re.compile(rb"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_CONFIG_SCOPES = frozenset({b"system", b"global", b"local", b"worktree", b"command"})
_REPOSITORY_CONFIG_SCOPES = frozenset({b"local", b"worktree"})


@dataclass(frozen = True)
class _CommandResult:
    code: int
    output: bytes
    overflowed: bool
    timed_out: bool


@dataclass(frozen = True)
class _Repository:
    root: Path
    identity: tuple[int, int]
    project_prefix: bytes


@dataclass(frozen = True)
class _StatusEntry:
    code: bytes
    path: bytes
    old_path: Optional[bytes]


@dataclass(frozen = True)
class _Capture:
    head: bytes
    branch: bytes
    status: bytes
    raw: bytes
    patch: bytes
    untracked: tuple[tuple[bytes, bytes], ...]
    config: bytes

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for label, payload in (
            (b"head", self.head),
            (b"branch", self.branch),
            (b"status", self.status),
            (b"raw", self.raw),
            (b"patch", self.patch),
        ):
            digest.update(label)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        for path, payload in self.untracked:
            digest.update(b"untracked")
            digest.update(len(path).to_bytes(8, "big"))
            digest.update(path)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()


def _identity(path: Path) -> tuple[int, int]:
    try:
        before = path.stat(follow_symlinks = False)
        resolved = path.resolve(strict = True)
        after = resolved.stat(follow_symlinks = False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AgentWorkspaceError("The project workspace identity is unavailable.") from exc
    if (
        not stat.S_ISDIR(before.st_mode)
        or not stat.S_ISDIR(after.st_mode)
        or (int(before.st_dev), int(before.st_ino)) != (int(after.st_dev), int(after.st_ino))
    ):
        raise AgentWorkspaceError("The project workspace identity changed.")
    return int(before.st_dev), int(before.st_ino)


def _assert_workspace_identity(workspace: ProjectWorkspace) -> None:
    current = _identity(workspace.root)
    expected = int(workspace.device_id), int(workspace.file_id)
    if current != expected:
        raise AgentWorkspaceError("The project workspace identity changed.")


def _windows_directory(csidl: int) -> Optional[Path]:
    if os.name != "nt":
        return None
    try:
        import ctypes

        buffer = ctypes.create_unicode_buffer(32_768)
        result = ctypes.windll.shell32.SHGetFolderPathW(None, csidl, None, 0, buffer)
        if result != 0 or not buffer.value:
            return None
        return Path(buffer.value)
    except (AttributeError, OSError, ValueError):
        return None


def _windows_system_root() -> Optional[Path]:
    if os.name != "nt":
        return None
    try:
        import ctypes

        buffer = ctypes.create_unicode_buffer(32_768)
        length = ctypes.windll.kernel32.GetWindowsDirectoryW(buffer, len(buffer))
        if length <= 0 or length >= len(buffer):
            return None
        return Path(buffer.value)
    except (AttributeError, OSError, ValueError):
        return None


def _within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
        return True
    except ValueError:
        return False


def _trusted_git_executable(
    *, _platform: Optional[str] = None, _windows_roots: Optional[Sequence[Path]] = None
) -> Path:
    """Resolve Git from fixed operating-system locations, never process PATH."""
    platform = _platform or os.name
    candidates: list[tuple[Path, Path]] = []
    if platform == "nt":
        roots = _windows_roots
        if roots is None:
            roots = tuple(
                root
                for root in (_windows_directory(0x26), _windows_directory(0x2A))
                if root is not None
            )
        for root in roots:
            candidates.extend(
                (root / "Git" / folder / "git.exe", root) for folder in ("cmd", "bin")
            )
    else:
        raw = shutil.which("git", path = os.defpath)
        if raw:
            path = Path(raw)
            candidates.append((path, path.parent))

    for raw, trusted_root in candidates:
        try:
            root = trusted_root.resolve(strict = True)
            candidate = raw.resolve(strict = True)
            metadata = candidate.stat(follow_symlinks = False)
        except (OSError, RuntimeError, ValueError):
            continue
        if not _within(candidate, root) or not stat.S_ISREG(metadata.st_mode):
            continue
        if platform != "nt" and not os.access(candidate, os.X_OK):
            continue
        return candidate
    raise AgentWorkspaceError("A trusted system Git executable is unavailable.")


def _kill_process(proc: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except OSError:
            pass


def _run_bounded(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    output_limit: int,
    timeout_seconds: float,
) -> _CommandResult:
    """Run a noninteractive command while draining but never retaining excess output."""
    if output_limit < 0 or timeout_seconds <= 0:
        raise ValueError("Invalid bounded process limits.")
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        proc = subprocess.Popen(
            list(command),
            cwd = cwd,
            env = env,
            stdin = subprocess.DEVNULL,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            shell = False,
            close_fds = True,
            start_new_session = os.name != "nt",
            creationflags = creationflags,
        )
    except OSError as exc:
        raise AgentWorkspaceError("The trusted Git process could not start.") from exc
    if proc.stdout is None or proc.stderr is None:
        _kill_process(proc)
        raise AgentWorkspaceError("The trusted Git process has incomplete output channels.")

    retained = bytearray()
    overflowed = threading.Event()
    stdout_finished = threading.Event()
    stderr_finished = threading.Event()
    observed_bytes = 0
    observed_lock = threading.Lock()

    def account_output(size: int) -> None:
        nonlocal observed_bytes
        with observed_lock:
            observed_bytes += size
            if observed_bytes > output_limit:
                overflowed.set()

    def read_output() -> None:
        try:
            while True:
                chunk = proc.stdout.read(65_536)
                if not chunk:
                    break
                account_output(len(chunk))
                available = output_limit - len(retained)
                if available > 0:
                    retained.extend(chunk[:available])
        finally:
            stdout_finished.set()

    def drain_stderr() -> None:
        try:
            while True:
                chunk = proc.stderr.read(65_536)
                if not chunk:
                    break
                account_output(len(chunk))
        finally:
            stderr_finished.set()

    stdout_reader = threading.Thread(target = read_output, daemon = True)
    stderr_reader = threading.Thread(target = drain_stderr, daemon = True)
    stdout_reader.start()
    stderr_reader.start()
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    while proc.poll() is None:
        if overflowed.is_set():
            _kill_process(proc)
            break
        if time.monotonic() >= deadline:
            timed_out = True
            _kill_process(proc)
            break
        time.sleep(0.01)
    try:
        proc.wait(timeout = 1)
    except subprocess.TimeoutExpired:
        _kill_process(proc)
    stdout_finished.wait(timeout = 1)
    stderr_finished.wait(timeout = 1)
    for channel in (proc.stdout, proc.stderr):
        try:
            channel.close()
        except OSError:
            pass
    return _CommandResult(
        code = int(proc.returncode if proc.returncode is not None else -1),
        output = bytes(retained[:output_limit]),
        overflowed = overflowed.is_set(),
        timed_out = timed_out,
    )


def _repository_scoped_config(raw: bytes) -> tuple[bytes, ...]:
    records = raw.split(b"\0")
    if records and records[-1] == b"":
        records.pop()
    if len(records) % 2:
        raise AgentWorkspaceError("Git returned an invalid configuration snapshot.")
    retained: list[bytes] = []
    for index in range(0, len(records), 2):
        scope = records[index]
        value = records[index + 1]
        if scope not in _CONFIG_SCOPES or not value:
            raise AgentWorkspaceError("Git returned an invalid configuration snapshot.")
        if scope in _REPOSITORY_CONFIG_SCOPES:
            retained.extend((scope, value))
    return tuple(retained)


class _GitSession:
    def __init__(self, executable: Path) -> None:
        self.executable = executable
        self._temporary: Optional[tempfile.TemporaryDirectory[str]] = None
        self.private_root: Optional[Path] = None
        self.hooks: Optional[Path] = None
        self.env: dict[str, str] = {}
        self._dynamic_overrides: list[str] = []

    def __enter__(self) -> "_GitSession":
        self._temporary = tempfile.TemporaryDirectory(prefix = "unsloth-git-review-")
        self.private_root = Path(self._temporary.name).resolve(strict = True)
        self.private_root.chmod(0o700)
        self.hooks = self.private_root / "hooks"
        self.hooks.mkdir(mode = 0o700)
        self.env = self._environment()
        return self

    def __exit__(self, _kind, _value, _traceback) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
        self._temporary = None
        self.private_root = None
        self.hooks = None
        self.env = {}
        self._dynamic_overrides = []

    def _environment(self) -> dict[str, str]:
        if self.private_root is None:
            raise RuntimeError("Git session is not active.")
        path_entries = [str(self.executable.parent)]
        env: dict[str, str] = {
            "HOME": str(self.private_root),
            "XDG_CONFIG_HOME": str(self.private_root),
            "PATH": os.pathsep.join(
                path_entries if os.name == "nt" else os.defpath.split(os.pathsep)
            ),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "",
            "GIT_EDITOR": "",
            "GIT_SEQUENCE_EDITOR": "",
            "GIT_MERGE_AUTOEDIT": "no",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GCM_INTERACTIVE": "Never",
            "PAGER": "",
            "EDITOR": "",
            "VISUAL": "",
            "LC_ALL": "C",
            "LANG": "C",
        }
        if os.name == "nt":
            system_root = _windows_system_root()
            if system_root is None:
                raise AgentWorkspaceError("The trusted Windows system root is unavailable.")
            env["SystemRoot"] = str(system_root)
            env["WINDIR"] = str(system_root)
            env["COMSPEC"] = str(system_root / "System32" / "cmd.exe")
            env["PATH"] = os.pathsep.join(
                [str(self.executable.parent), str(system_root / "System32"), str(system_root)]
            )
        env["TMPDIR" if os.name != "nt" else "TEMP"] = str(self.private_root)
        env["TMP"] = str(self.private_root)
        return env

    def _base(self) -> list[str]:
        if self.hooks is None:
            raise RuntimeError("Git session is not active.")
        settings = (
            f"core.hooksPath={self.hooks}",
            "core.fsmonitor=false",
            "core.untrackedCache=false",
            "core.attributesFile=" + os.devnull,
            "core.excludesFile=" + os.devnull,
            "core.quotePath=true",
            "status.relativePaths=false",
            "color.ui=false",
            "core.pager=",
            "credential.helper=",
            "credential.interactive=never",
            "core.askPass=",
            "commit.gpgSign=false",
            "tag.gpgSign=false",
            "merge.verifySignatures=false",
            "log.showSignature=false",
            "gpg.program=",
            "gpg.ssh.program=",
            "diff.external=",
            "diff.orderFile=" + os.devnull,
            "interactive.diffFilter=",
            "core.editor=",
            "sequence.editor=",
            "merge.autoEdit=no",
            "merge.renormalize=false",
            "core.sshCommand=false",
            "core.gitProxy=false",
            "core.symlinks=false",
            "core.protectNTFS=true",
            "core.protectHFS=true",
            "maintenance.auto=false",
            "gc.auto=0",
            "gc.autoDetach=false",
            "fetch.autoMaintenance=false",
            "submodule.recurse=false",
            "protocol.allow=never",
        )
        command = [str(self.executable), "--no-pager"]
        for setting in settings:
            command.extend(("-c", setting))
        command.extend(self._dynamic_overrides)
        return command

    def run(
        self,
        cwd: Path,
        args: Sequence[str],
        *,
        output_limit: int,
        timeout_seconds: float = 15,
        allowed_codes: frozenset[int] = frozenset({0}),
    ) -> _CommandResult:
        result = _run_bounded(
            [*self._base(), *args],
            cwd = cwd,
            env = self.env,
            output_limit = output_limit,
            timeout_seconds = timeout_seconds,
        )
        if result.timed_out:
            raise AgentWorkspaceError("The Git review command timed out.")
        if result.overflowed:
            raise OverflowError("git-output-limit")
        if result.code not in allowed_codes:
            raise AgentWorkspaceError("Git could not read the project repository safely.")
        return result

    def inspect_executable_config(self, repository: Path) -> bytes:
        """Snapshot repository config and override every executable diff/filter driver."""
        result = self.run(
            repository,
            ("config", "--includes", "--show-scope", "--null", "--list"),
            output_limit = MAX_CONFIG_BYTES,
            timeout_seconds = 5,
            allowed_codes = frozenset({0, 1}),
        )
        names = self.run(
            repository,
            (
                "config",
                "--includes",
                "--show-scope",
                "--null",
                "--name-only",
                "--get-regexp",
                r"^(filter\..*\.(clean|smudge|process|required)|diff\..*\.(command|textconv)|merge\..*\.driver)$",
            ),
            output_limit = MAX_CONFIG_BYTES,
            timeout_seconds = 5,
            allowed_codes = frozenset({0, 1}),
        )
        drivers: set[tuple[bytes, bytes]] = set()
        repository_names = _repository_scoped_config(names.output)
        for raw_key in repository_names[1::2]:
            match = _DRIVER_KEY.fullmatch(raw_key)
            if match is None:
                raise AgentWorkspaceError("Git executable-driver configuration is invalid.")
            drivers.add((match.group("kind"), match.group("driver")))
            if len(drivers) > MAX_CONFIG_DRIVERS:
                raise OverflowError("git-config-driver-limit")
        overrides: list[str] = []
        for kind, raw_driver in sorted(drivers):
            driver = raw_driver.decode("ascii")
            if kind == b"filter":
                for option, value in (
                    ("process", ""),
                    ("clean", ""),
                    ("smudge", ""),
                    ("required", "false"),
                ):
                    overrides.extend(("-c", f"filter.{driver}.{option}={value}"))
            elif kind == b"merge":
                overrides.extend(("-c", f"merge.{driver}.driver=false"))
            else:
                overrides.extend(("-c", f"diff.{driver}.command="))
                overrides.extend(("-c", f"diff.{driver}.textconv="))
        self._dynamic_overrides = overrides
        snapshot = _repository_scoped_config(result.output)
        return b"\0".join(snapshot) + (b"\0" if snapshot else b"")


def _discover_repository(session: _GitSession, workspace: ProjectWorkspace) -> _Repository:
    result = session.run(
        workspace.root,
        ("rev-parse", "--show-toplevel"),
        output_limit = 32_768,
        timeout_seconds = 5,
    )
    raw = result.output.rstrip(b"\r\n")
    if not raw or b"\0" in raw:
        raise AgentWorkspaceError("Git returned an invalid repository root.")
    try:
        repository = Path(os.fsdecode(raw)).resolve(strict = True)
        project = workspace.root.resolve(strict = True)
        relative = project.relative_to(repository)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AgentWorkspaceError("The project folder is not inside its Git repository.") from exc
    repo_identity = _identity(repository)
    prefix = os.fsencode(relative.as_posix())
    if prefix == b".":
        prefix = b""
    return _Repository(repository, repo_identity, prefix)


def _assert_repository_identity(repository: _Repository) -> None:
    if _identity(repository.root) != repository.identity:
        raise AgentWorkspaceError("The Git repository identity changed.")


def _pathspec(repository: _Repository) -> list[str]:
    if not repository.project_prefix:
        return []
    try:
        decoded = repository.project_prefix.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AgentWorkspaceError("The project subdirectory is not valid UTF-8.") from exc
    return [f":(top,literal){decoded}"]


def _project_path(repository: _Repository, path: bytes) -> Optional[bytes]:
    if not path or b"\0" in path or path.startswith((b"/", b"\\")):
        raise AgentWorkspaceError("Git returned an invalid project path.")
    if any(part in {b"", b".", b".."} for part in path.replace(b"\\", b"/").split(b"/")):
        raise AgentWorkspaceError("Git returned an invalid project path.")
    if not repository.project_prefix:
        return path
    prefix = repository.project_prefix + b"/"
    if path.startswith(prefix):
        return path[len(prefix) :]
    return None


def _visible_text(value: str, *, escape_backslash: bool = False) -> str:
    visible: list[str] = []
    for character in value:
        code = ord(character)
        if character == "\\" and escape_backslash:
            visible.append("\\\\")
        elif character == "\n":
            visible.append("\\n")
        elif character == "\r":
            visible.append("\\r")
        elif character == "\t":
            visible.append("\\t")
        elif code < 0x20 or 0x7F <= code <= 0x9F or unicodedata.category(character) == "Cf":
            visible.append(f"\\u{code:04X}" if code <= 0xFFFF else f"\\u{{{code:X}}}")
        else:
            visible.append(character)
    return "".join(visible)


def _visible_path(path: Optional[bytes]) -> tuple[Optional[str], Optional[str]]:
    if path is None:
        return None, None
    try:
        decoded = path.decode("utf-8")
        return _visible_text(decoded, escape_backslash = True), "utf-8"
    except UnicodeDecodeError:
        pieces: list[str] = []
        index = 0
        while index < len(path):
            try:
                decoded = path[index:].decode("utf-8")
                pieces.append(_visible_text(decoded, escape_backslash = True))
                break
            except UnicodeDecodeError as exc:
                if exc.start:
                    pieces.append(
                        _visible_text(
                            path[index : index + exc.start].decode("utf-8"),
                            escape_backslash = True,
                        )
                    )
                bad_start = index + exc.start
                bad_end = index + max(exc.end, exc.start + 1)
                pieces.extend(f"\\x{byte:02X}" for byte in path[bad_start:bad_end])
                index = bad_end
        return "".join(pieces), "escaped"


def _stable_id(kind: bytes, *parts: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(str(MANIFEST_VERSION).encode("ascii"))
    digest.update(b"\0")
    digest.update(kind)
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _parse_status(repository: _Repository, raw: bytes) -> list[_StatusEntry]:
    records = raw.split(b"\0")
    if records and records[-1] == b"":
        records.pop()
    entries: list[_StatusEntry] = []
    index = 0
    while index < len(records):
        record = records[index]
        if len(record) < 4 or record[2:3] != b" ":
            raise AgentWorkspaceError("Git returned an invalid status manifest.")
        code = record[:2]
        path = record[3:]
        old_path: Optional[bytes] = None
        index += 1
        if code[:1] in {b"R", b"C"} or code[1:2] in {b"R", b"C"}:
            if index >= len(records):
                raise AgentWorkspaceError("Git returned an invalid rename status.")
            old_path = records[index]
            index += 1
        local_path = _project_path(repository, path)
        local_old = _project_path(repository, old_path) if old_path is not None else None
        if local_path is None and local_old is None:
            continue
        normalized_code = code
        normalized_old = local_old
        if old_path is not None and (local_path is None or local_old is None):
            replacement = b"A" if local_path is not None else b"D"
            normalized_code = bytes(replacement[0] if value in b"RC" else value for value in code)
            normalized_old = None
        entries.append(
            _StatusEntry(
                code = normalized_code,
                path = local_path if local_path is not None else local_old or b"",
                old_path = normalized_old,
            )
        )
        if len(entries) > MAX_FILES:
            raise OverflowError("file-limit")
    return entries


def _status_command(repository: _Repository) -> list[str]:
    return [
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=dirty",
        "--",
        *_pathspec(repository),
    ]


def _head_and_branch(session: _GitSession, repository: _Repository) -> tuple[bytes, bytes]:
    head = session.run(
        repository.root,
        ("rev-parse", "--verify", "--quiet", "HEAD^{commit}"),
        output_limit = 256,
        timeout_seconds = 5,
        allowed_codes = frozenset({0, 1}),
    ).output.strip()
    if head and _OBJECT_ID.fullmatch(head) is None:
        raise AgentWorkspaceError("Git returned an invalid HEAD object ID.")
    branch_result = session.run(
        repository.root,
        ("symbolic-ref", "--quiet", "--short", "HEAD"),
        output_limit = 4_096,
        timeout_seconds = 5,
        allowed_codes = frozenset({0, 1}),
    )
    branch = branch_result.output.rstrip(b"\r\n")
    if b"\0" in branch:
        raise AgentWorkspaceError("Git returned an invalid branch name.")
    return head, branch


def _diff_arguments(session: _GitSession, repository: _Repository, mode: str) -> list[str]:
    if mode == "staged":
        return ["--cached"]
    if mode == "unstaged":
        return []
    head, _branch = _head_and_branch(session, repository)
    if head:
        return [head.decode("ascii")]
    empty = session.run(
        repository.root,
        ("hash-object", "-t", "tree", "--stdin"),
        output_limit = 256,
        timeout_seconds = 5,
    ).output.strip()
    if _OBJECT_ID.fullmatch(empty) is None:
        raise AgentWorkspaceError("Git returned an invalid empty-tree object ID.")
    return [empty.decode("ascii")]


def _untracked_evidence(
    workspace: ProjectWorkspace, entries: Sequence[_StatusEntry], *, content_limit: int
) -> tuple[tuple[bytes, bytes], ...]:
    evidence: list[tuple[bytes, bytes]] = []
    remaining = content_limit
    for entry in entries:
        if entry.code != b"??":
            continue
        read_limit = min(MAX_UNTRACKED_FILE_BYTES, max(0, remaining))
        if read_limit == 0:
            evidence.append((entry.path, b"limit"))
            continue
        try:
            target = os.fsdecode(entry.path)
            content, mode, identity = read_project_file(workspace, target, read_limit)
            payload = (
                b"content\0"
                + str(mode).encode("ascii")
                + b"\0"
                + str(identity[0]).encode("ascii")
                + b":"
                + str(identity[1]).encode("ascii")
                + b"\0"
                + content
            )
            remaining -= len(content)
        except OverflowError:
            payload = b"oversize"
        except OSError as exc:
            payload = (
                b"symlink-or-reparse"
                if exc.errno in {errno.ELOOP, errno.EMLINK}
                else b"unsafe-or-unavailable"
            )
        except (AgentWorkspaceError, RuntimeError, ValueError):
            payload = b"unsafe-or-unavailable"
        evidence.append((entry.path, payload))
    return tuple(evidence)


def _capture(
    session: _GitSession,
    workspace: ProjectWorkspace,
    repository: _Repository,
    *,
    mode: Optional[str],
    max_bytes: int,
) -> _Capture:
    config_before = session.inspect_executable_config(repository.root)
    head, branch = _head_and_branch(session, repository)
    status = session.run(
        repository.root,
        _status_command(repository),
        output_limit = MAX_STATUS_BYTES,
        timeout_seconds = 20,
    ).output
    entries = _parse_status(repository, status)
    raw = b""
    patch = b""
    if mode is not None and not any(entry.code in _CONFLICT_CODES for entry in entries):
        mode_args = _diff_arguments(session, repository, mode)
        common = [
            *mode_args,
            "--full-index",
            "--find-renames",
            "--find-copies-harder",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=dirty",
            "--",
            *_pathspec(repository),
        ]
        raw = session.run(
            repository.root,
            ("diff", "--raw", "-z", "--no-abbrev", *common),
            output_limit = min(MAX_STATUS_BYTES, max_bytes),
            timeout_seconds = 20,
        ).output
        patch = session.run(
            repository.root,
            (
                "diff",
                "--patch",
                "--binary",
                "--no-color",
                "--unified=3",
                *common,
            ),
            output_limit = max_bytes,
            timeout_seconds = 20,
        ).output
    untracked = _untracked_evidence(
        workspace,
        entries if mode in {"head", "unstaged"} else (),
        content_limit = max_bytes,
    )
    config_after = session.inspect_executable_config(repository.root)
    if config_after != config_before:
        raise AgentWorkspaceError("Git configuration changed during review.")
    return _Capture(head, branch, status, raw, patch, untracked, config_before)


def _coherent_capture(
    session: _GitSession,
    workspace: ProjectWorkspace,
    repository: _Repository,
    *,
    mode: Optional[str],
    max_bytes: int,
) -> tuple[_Capture, bool]:
    _assert_workspace_identity(workspace)
    _assert_repository_identity(repository)
    first = _capture(
        session,
        workspace,
        repository,
        mode = mode,
        max_bytes = max_bytes,
    )
    _assert_workspace_identity(workspace)
    _assert_repository_identity(repository)
    second = _capture(
        session,
        workspace,
        repository,
        mode = mode,
        max_bytes = max_bytes,
    )
    _assert_workspace_identity(workspace)
    _assert_repository_identity(repository)
    return first, first == second


def _status_public(
    project_id: str,
    workspace: ProjectWorkspace,
    capture: _Capture,
    entries: Sequence[_StatusEntry],
    *,
    coherent: bool,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    counts = {"staged": 0, "unstaged": 0, "untracked": 0, "conflicted": 0}
    for entry in entries:
        index_code = chr(entry.code[0])
        worktree_code = chr(entry.code[1])
        conflicted = entry.code in _CONFLICT_CODES
        if entry.code == b"??":
            counts["untracked"] += 1
        else:
            if index_code not in {" ", "?"}:
                counts["staged"] += 1
            if worktree_code not in {" ", "?"}:
                counts["unstaged"] += 1
        if conflicted:
            counts["conflicted"] += 1
        path, encoding = _visible_path(entry.path)
        old_path, old_encoding = _visible_path(entry.old_path)
        files.append(
            {
                "id": _stable_id(b"status", entry.code, entry.path, entry.old_path or b""),
                "code": entry.code.decode("ascii"),
                "indexStatus": index_code,
                "worktreeStatus": worktree_code,
                "path": path,
                "pathEncoding": encoding,
                "oldPath": old_path,
                "oldPathEncoding": old_encoding,
                "conflicted": conflicted,
            }
        )
    head = capture.head.decode("ascii") if capture.head else None
    branch = capture.branch.decode("utf-8", errors = "backslashreplace") or None
    return {
        "version": MANIFEST_VERSION,
        "projectId": project_id,
        "target": {"kind": "primary"},
        "workspaceRevision": int(workspace.revision),
        "head": head,
        "branch": (_visible_text(branch, escape_backslash = True) if branch is not None else None),
        "sourceFingerprint": capture.fingerprint(),
        "fingerprintComplete": False,
        "coherent": coherent,
        "blockedReasons": [] if coherent else ["workspace-changed-during-review"],
        "counts": counts,
        "files": files if coherent else [],
    }


def git_status(project_id: str) -> dict[str, Any]:
    """Return a coherent read-only status snapshot for the primary project workspace."""
    with project_workspace_access(project_id) as workspace:
        executable = _trusted_git_executable()
        with _GitSession(executable) as session:
            repository = _discover_repository(session, workspace)
            capture, coherent = _coherent_capture(
                session,
                workspace,
                repository,
                mode = None,
                max_bytes = DEFAULT_MAX_BYTES,
            )
        entries = _parse_status(repository, capture.status)
        return _status_public(project_id, workspace, capture, entries, coherent = coherent)


def _parse_raw(repository: _Repository, raw: bytes) -> list[dict[str, Any]]:
    tokens = raw.split(b"\0")
    if tokens and tokens[-1] == b"":
        tokens.pop()
    entries: list[dict[str, Any]] = []
    index = 0
    while index < len(tokens):
        match = _RAW_HEADER.fullmatch(tokens[index])
        if match is None or index + 1 >= len(tokens):
            raise AgentWorkspaceError("Git returned an invalid raw diff manifest.")
        status = match.group("status")
        first_path = tokens[index + 1]
        index += 2
        old_repo_path: Optional[bytes] = None
        repo_path = first_path
        if status.startswith((b"R", b"C")):
            if index >= len(tokens):
                raise AgentWorkspaceError("Git returned an invalid rename diff manifest.")
            old_repo_path = first_path
            repo_path = tokens[index]
            index += 1
        path = _project_path(repository, repo_path)
        old_path = _project_path(repository, old_repo_path) if old_repo_path else None
        if path is None and old_path is None:
            continue
        scope_boundary = old_repo_path is not None and (path is None or old_path is None)
        if scope_boundary:
            object_id_size = len(match.group("old_sha"))
            if path is not None:
                status = b"A"
                old_path = None
                old_mode = b"000000"
                new_mode = match.group("new_mode")
                old_blob = b"0" * object_id_size
                new_blob = match.group("new_sha")
            else:
                status = b"D"
                path = old_path
                old_path = None
                old_mode = match.group("old_mode")
                new_mode = b"000000"
                old_blob = match.group("old_sha")
                new_blob = b"0" * object_id_size
        else:
            old_mode = match.group("old_mode")
            new_mode = match.group("new_mode")
            old_blob = match.group("old_sha")
            new_blob = match.group("new_sha")
        entries.append(
            {
                "code": status,
                "path": path,
                "oldPath": old_path,
                "oldMode": old_mode,
                "newMode": new_mode,
                "oldBlob": old_blob,
                "newBlob": new_blob,
                "scopeBoundary": scope_boundary,
            }
        )
        if len(entries) > MAX_FILES:
            raise OverflowError("file-limit")
    return entries


def _patch_sections(raw: bytes) -> list[bytes]:
    if not raw:
        return []
    starts = [match.start() for match in re.finditer(rb"(?m)^diff --git ", raw)]
    if not starts or starts[0] != 0:
        raise AgentWorkspaceError("Git returned an invalid patch manifest.")
    return [
        raw[start : starts[index + 1] if index + 1 < len(starts) else len(raw)]
        for index, start in enumerate(starts)
    ]


def _structured_hunks(section: str) -> tuple[list[dict[str, Any]], int, int, int]:
    raw_lines = section.split("\n")
    hunks: list[dict[str, Any]] = []
    additions = 0
    deletions = 0
    line_count = 0
    index = 0
    while index < len(raw_lines):
        match = _HUNK_HEADER.match(raw_lines[index])
        if match is None:
            index += 1
            continue
        header = raw_lines[index]
        old_line = int(match.group("old_start"))
        new_line = int(match.group("new_start"))
        expected_old = int(match.group("old_lines") or "1")
        expected_new = int(match.group("new_lines") or "1")
        old_seen = 0
        new_seen = 0
        canonical: list[str] = []
        lines: list[dict[str, Any]] = []
        index += 1
        while index < len(raw_lines) and not raw_lines[index].startswith("@@ "):
            raw_line = raw_lines[index]
            if raw_line.startswith("diff --git "):
                break
            if raw_line == r"\ No newline at end of file":
                if not lines:
                    raise AgentWorkspaceError("Git returned an invalid no-newline marker.")
                lines[-1]["noNewline"] = True
                canonical.append(raw_line)
                index += 1
                continue
            if not raw_line or raw_line[0] not in {" ", "+", "-"}:
                break
            text = raw_line[1:]
            if len(text) > MAX_LINE_CHARS:
                raise OverflowError("line-size-limit")
            if raw_line[0] == " ":
                lines.append(
                    {
                        "kind": "context",
                        "text": _visible_text(text),
                        "oldLine": old_line,
                        "newLine": new_line,
                    }
                )
                old_line += 1
                new_line += 1
                old_seen += 1
                new_seen += 1
            elif raw_line[0] == "+":
                lines.append(
                    {
                        "kind": "add",
                        "text": _visible_text(text),
                        "oldLine": None,
                        "newLine": new_line,
                    }
                )
                additions += 1
                new_line += 1
                new_seen += 1
            else:
                lines.append(
                    {
                        "kind": "delete",
                        "text": _visible_text(text),
                        "oldLine": old_line,
                        "newLine": None,
                    }
                )
                deletions += 1
                old_line += 1
                old_seen += 1
            canonical.append(raw_line)
            line_count += 1
            if line_count > MAX_LINES:
                raise OverflowError("line-limit")
            index += 1
        if old_seen != expected_old or new_seen != expected_new:
            raise AgentWorkspaceError("Git returned an inconsistent diff hunk.")
        hunks.append(
            {
                "header": _visible_text(header),
                "oldStart": int(match.group("old_start")),
                "oldLines": expected_old,
                "newStart": int(match.group("new_start")),
                "newLines": expected_new,
                "lines": lines,
                "_canonical": "\n".join(canonical).encode("utf-8"),
            }
        )
        if len(hunks) > MAX_HUNKS:
            raise OverflowError("hunk-limit")
    return hunks, additions, deletions, line_count


def _file_manifest(entry: dict[str, Any], section: bytes, *, mode: str) -> tuple[dict, int, int]:
    path: bytes = entry["path"]
    old_path: Optional[bytes] = entry["oldPath"]
    code: bytes = entry["code"]
    binary = b"\nBinary files " in b"\n" + section or b"\nGIT binary patch\n" in b"\n" + section
    encoding = "utf-8"
    try:
        decoded = section.decode("utf-8")
    except UnicodeDecodeError:
        decoded = ""
        encoding = "invalid-utf8"
    hunks: list[dict[str, Any]] = []
    additions = 0
    deletions = 0
    line_count = 0
    scope_boundary = bool(entry.get("scopeBoundary"))
    if decoded and not binary and not scope_boundary:
        hunks, additions, deletions, line_count = _structured_hunks(decoded)
    old_mode: bytes = entry["oldMode"]
    new_mode: bytes = entry["newMode"]
    special_mode = b"160000" in {old_mode, new_mode} or b"120000" in {old_mode, new_mode}
    mode_changed = old_mode not in {b"000000", new_mode} and new_mode != b"000000"
    whole_file_only = (
        binary
        or encoding != "utf-8"
        or code[:1] in {b"R", b"C", b"T", b"U"}
        or scope_boundary
        or special_mode
        or mode_changed
        or not hunks
    )
    public_hunks: list[dict[str, Any]] = []
    for hunk in hunks:
        canonical = hunk.pop("_canonical")
        hunk["id"] = _stable_id(
            b"hunk", mode.encode("ascii"), path, hunk["header"].encode("utf-8"), canonical
        )
        public_hunks.append(hunk)
    visible_path, path_encoding = _visible_path(path)
    visible_old, old_path_encoding = _visible_path(old_path)
    return (
        {
            "id": _stable_id(b"file", mode.encode("ascii"), code, path, old_path or b""),
            "code": code.decode("ascii"),
            "path": visible_path,
            "pathEncoding": path_encoding,
            "oldPath": visible_old,
            "oldPathEncoding": old_path_encoding,
            "oldMode": old_mode.decode("ascii"),
            "newMode": new_mode.decode("ascii"),
            "oldBlob": entry["oldBlob"].decode("ascii"),
            "newBlob": entry["newBlob"].decode("ascii"),
            "binary": binary,
            "encoding": encoding,
            "symlink": b"120000" in {old_mode, new_mode},
            "submodule": b"160000" in {old_mode, new_mode},
            "modeChanged": mode_changed,
            "scopeBoundary": scope_boundary,
            "wholeFileOnly": whole_file_only,
            "truncated": False,
            "additions": additions,
            "deletions": deletions,
            "hunks": [] if whole_file_only else public_hunks,
        },
        0 if whole_file_only else len(public_hunks),
        line_count,
    )


def _untracked_manifest(path: bytes, payload: bytes, *, mode: str) -> tuple[dict, int, int]:
    visible_path, path_encoding = _visible_path(path)
    parts = payload.split(b"\0", 3) if payload.startswith(b"content\0") else []
    content = parts[3] if len(parts) == 4 else b""
    raw_mode = parts[1] if len(parts) == 4 else b""
    try:
        new_mode = f"{int(raw_mode):06o}" if raw_mode else "000000"
    except ValueError:
        new_mode = "000000"
    base = {
        "id": _stable_id(b"file", mode.encode("ascii"), b"??", path),
        "code": "??",
        "path": visible_path,
        "pathEncoding": path_encoding,
        "oldPath": None,
        "oldPathEncoding": None,
        "oldMode": "000000",
        "newMode": new_mode,
        "oldBlob": "0" * 40,
        "newBlob": "0" * 40,
        "symlink": payload == b"symlink-or-reparse",
        "submodule": False,
        "modeChanged": False,
        "truncated": payload in {b"limit", b"oversize"},
        "byteSize": len(content) if len(parts) == 4 else None,
        "unavailableReason": (
            payload.decode("ascii") if not payload.startswith(b"content\0") else None
        ),
    }
    if not payload.startswith(b"content\0"):
        return (
            base
            | {
                "binary": False,
                "encoding": "unavailable",
                "wholeFileOnly": True,
                "additions": 0,
                "deletions": 0,
                "hunks": [],
            },
            0,
            0,
        )
    binary = b"\0" in content
    try:
        decoded = content.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        decoded = ""
        encoding = "invalid-utf8"
    if binary or encoding != "utf-8":
        return (
            base
            | {
                "binary": binary,
                "encoding": encoding,
                "wholeFileOnly": True,
                "additions": 0,
                "deletions": 0,
                "hunks": [],
            },
            0,
            0,
        )
    raw_lines = decoded.splitlines()
    if len(raw_lines) > MAX_LINES or any(len(line) > MAX_LINE_CHARS for line in raw_lines):
        return (
            base
            | {
                "binary": False,
                "encoding": "utf-8",
                "wholeFileOnly": True,
                "truncated": True,
                "additions": 0,
                "deletions": 0,
                "hunks": [],
            },
            0,
            0,
        )
    lines = [
        {
            "kind": "add",
            "text": _visible_text(line),
            "oldLine": None,
            "newLine": index + 1,
        }
        for index, line in enumerate(raw_lines)
    ]
    header = f"@@ -0,0 +1,{len(lines)} @@"
    hunk = {
        "id": _stable_id(b"hunk", mode.encode("ascii"), path, header.encode("ascii"), content),
        "header": header,
        "oldStart": 0,
        "oldLines": 0,
        "newStart": 1,
        "newLines": len(lines),
        "lines": lines,
    }
    return (
        base
        | {
            "binary": False,
            "encoding": "utf-8",
            "wholeFileOnly": False,
            "additions": len(lines),
            "deletions": 0,
            "hunks": [hunk],
        },
        1,
        len(lines),
    )


def _blocked_manifest(
    project_id: str,
    workspace: ProjectWorkspace,
    capture: _Capture,
    *,
    mode: str,
    reasons: list[str],
    conflicts: list[str],
    truncated: bool,
    max_bytes: int,
) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "projectId": project_id,
        "target": {"kind": "primary"},
        "workspaceRevision": int(workspace.revision),
        "mode": mode,
        "head": capture.head.decode("ascii") if capture.head else None,
        "sourceFingerprint": capture.fingerprint(),
        "fingerprintComplete": False,
        "selectable": False,
        "blockedReasons": reasons,
        "conflictedPaths": conflicts,
        "files": [],
        "fileCount": 0,
        "hunkCount": 0,
        "lineCount": 0,
        "truncated": truncated,
        "limits": _limits(max_bytes),
    }


def _limits(max_bytes: int) -> dict[str, int]:
    return {
        "maxBytes": max_bytes,
        "maxFiles": MAX_FILES,
        "maxHunks": MAX_HUNKS,
        "maxLines": MAX_LINES,
        "maxLineChars": MAX_LINE_CHARS,
        "maxUntrackedFileBytes": MAX_UNTRACKED_FILE_BYTES,
    }


def build_diff_manifest(
    project_id: str,
    *,
    mode: str = "head",
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    """Return a structured, read-only diff for the primary project workspace."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in DIFF_MODES:
        raise AgentWorkspaceError("Diff review mode must be head, staged, or unstaged.")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise AgentWorkspaceError("Diff review byte limit is invalid.")
    bounded_bytes = max(4_096, min(max_bytes, MAX_MAX_BYTES))
    with project_workspace_access(project_id) as workspace:
        executable = _trusted_git_executable()
        try:
            with _GitSession(executable) as session:
                repository = _discover_repository(session, workspace)
                capture, coherent = _coherent_capture(
                    session,
                    workspace,
                    repository,
                    mode = normalized_mode,
                    max_bytes = bounded_bytes,
                )
        except OverflowError as exc:
            empty = _Capture(b"", b"", b"", b"", b"", (), b"")
            return _blocked_manifest(
                project_id,
                workspace,
                empty,
                mode = normalized_mode,
                reasons = [str(exc)],
                conflicts = [],
                truncated = True,
                max_bytes = bounded_bytes,
            )
        entries = _parse_status(repository, capture.status)
        conflicts = [
            _visible_path(entry.path)[0] or "" for entry in entries if entry.code in _CONFLICT_CODES
        ]
        if not coherent or conflicts:
            reasons = []
            if not coherent:
                reasons.append("workspace-changed-during-review")
            if conflicts:
                reasons.append("repository-conflicts")
            return _blocked_manifest(
                project_id,
                workspace,
                capture,
                mode = normalized_mode,
                reasons = reasons,
                conflicts = conflicts,
                truncated = False,
                max_bytes = bounded_bytes,
            )
        try:
            raw_entries = _parse_raw(repository, capture.raw)
            sections = _patch_sections(capture.patch)
            if len(raw_entries) != len(sections):
                raise AgentWorkspaceError("Git diff metadata did not match its patch output.")
            files: list[dict[str, Any]] = []
            hunk_count = 0
            line_count = 0
            for entry, section in zip(raw_entries, sections):
                file, file_hunks, file_lines = _file_manifest(
                    entry,
                    section,
                    mode = normalized_mode,
                )
                files.append(file)
                hunk_count += file_hunks
                line_count += file_lines
            if normalized_mode in {"head", "unstaged"}:
                seen = {entry["path"] for entry in raw_entries}
                for path, payload in capture.untracked:
                    if path in seen:
                        continue
                    file, file_hunks, file_lines = _untracked_manifest(
                        path,
                        payload,
                        mode = normalized_mode,
                    )
                    files.append(file)
                    hunk_count += file_hunks
                    line_count += file_lines
            if len(files) > MAX_FILES:
                raise OverflowError("file-limit")
            if hunk_count > MAX_HUNKS:
                raise OverflowError("hunk-limit")
            if line_count > MAX_LINES:
                raise OverflowError("line-limit")
        except (AgentWorkspaceError, OverflowError) as exc:
            reason = str(exc) if isinstance(exc, OverflowError) else "diff-parse-invalid"
            return _blocked_manifest(
                project_id,
                workspace,
                capture,
                mode = normalized_mode,
                reasons = [reason],
                conflicts = [],
                truncated = isinstance(exc, OverflowError),
                max_bytes = bounded_bytes,
            )
        fingerprint_complete = all(
            payload.startswith(b"content\0") for _path, payload in capture.untracked
        )
        blocked_reasons = [] if fingerprint_complete else ["untracked-content-incomplete"]
        return {
            "version": MANIFEST_VERSION,
            "projectId": project_id,
            "target": {"kind": "primary"},
            "workspaceRevision": int(workspace.revision),
            "mode": normalized_mode,
            "head": capture.head.decode("ascii") if capture.head else None,
            "sourceFingerprint": capture.fingerprint(),
            "fingerprintComplete": fingerprint_complete,
            "selectable": fingerprint_complete,
            "blockedReasons": blocked_reasons,
            "conflictedPaths": [],
            "files": files,
            "fileCount": len(files),
            "hunkCount": hunk_count,
            "lineCount": line_count,
            "truncated": False,
            "limits": _limits(bounded_bytes),
        }


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DIFF_MODES",
    "MAX_MAX_BYTES",
    "build_diff_manifest",
    "git_status",
]
