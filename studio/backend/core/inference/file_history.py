# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-chat file history for /rewind; index work always on the tool's worker thread, never the event
loop; processes and packages not restored"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import stat as stat_mod
import threading
import time
import uuid
import zlib
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

try:
    import zstandard as _zstandard
    _ZSTD_C = _zstandard.ZstdCompressor(level = 1)
    _ZSTD_D = _zstandard.ZstdDecompressor()
except ImportError:
    _ZSTD_C = _ZSTD_D = None

_RAW, _ZLIB, _ZSTD = b"R", b"Z", b"S"

MAX_FILE_BYTES = 16 << 20
BASELINE_BUDGET_BYTES = 256 << 20
CALL_BUDGET_BYTES = 256 << 20
CHAT_CAP_CHECKPOINTS = 500
GLOBAL_CAP_BYTES = 2 << 30
GC_AFTER_BYTES = 128 << 20
# Studio's own trees inside a chat folder: content addressed copies, the scratch dir, spilled output.
_SKIP_PREFIXES = (".unsloth_attachments/", "unsloth-tmp/", ".unsloth_tool_output/")

# (thread_id, user_message_id) of the turn being served, set once per chat request.
_TURN: ContextVar = ContextVar("unsloth_file_history_turn", default = None)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS blobs(
  hash TEXT PRIMARY KEY, size INTEGER NOT NULL, stored INTEGER NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS checkpoints(
  id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT, workdir TEXT NOT NULL, turn_id TEXT,
  kind TEXT NOT NULL, ambiguous INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS ck_chat ON checkpoints(chat_id, turn_id);
CREATE TABLE IF NOT EXISTS file_versions(
  checkpoint_id INTEGER NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
  rel_path TEXT NOT NULL, before_kind TEXT NOT NULL, before_blob TEXT, before_mode INTEGER,
  after_kind TEXT NOT NULL, after_blob TEXT, after_mode INTEGER,
  PRIMARY KEY(checkpoint_id, rel_path));
CREATE INDEX IF NOT EXISTS fv_before ON file_versions(before_blob);
CREATE INDEX IF NOT EXISTS fv_after ON file_versions(after_blob);
CREATE TABLE IF NOT EXISTS workdir_state(
  workdir TEXT NOT NULL, rel_path TEXT NOT NULL, mtime_ns INTEGER, size INTEGER, digest TEXT,
  kind TEXT NOT NULL, blob TEXT, mode INTEGER, PRIMARY KEY(workdir, rel_path));
CREATE INDEX IF NOT EXISTS ws_blob ON workdir_state(blob);
CREATE TABLE IF NOT EXISTS workdirs(workdir TEXT PRIMARY KEY, baseline_at REAL);
CREATE TABLE IF NOT EXISTS chat_workdirs(chat_id TEXT NOT NULL, workdir TEXT NOT NULL,
  PRIMARY KEY(chat_id, workdir));
"""


class RewindBusy(RuntimeError):
    """A tool call is running in the folder, or another restore is."""


@dataclass
class _Entry:
    mtime_ns: int
    size: int
    digest: Optional[str]
    kind: str  # "blob" or "untracked" (over the size cap or budget, or unreadable)
    blob: Optional[str]
    mode: Optional[int]


@dataclass
class CallToken:
    chat_id: str
    workdir: str
    turn_id: str
    ambiguous: bool = False
    done: bool = False


@dataclass
class RewindReport:
    has_changes: bool = False
    restored: list = field(default_factory = list)
    deleted: list = field(default_factory = list)
    recreated: list = field(default_factory = list)
    conflicts: list = field(default_factory = list)
    untracked: list = field(default_factory = list)
    errors: list = field(default_factory = list)
    undo_turn_id: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "has_changes": self.has_changes,
            "restored": self.restored,
            "deleted": self.deleted,
            "recreated": self.recreated,
            "conflicts": self.conflicts,
            "untracked": self.untracked,
            "errors": self.errors,
            "undo_turn_id": self.undo_turn_id,
        }


def set_turn(thread_id: Optional[str], user_message_id: Optional[str]) -> None:
    """Record which chat turn this request serves. Without both ids nothing is captured."""
    _TURN.set((thread_id, user_message_id) if thread_id and user_message_id else None)


def current_turn() -> Optional[tuple]:
    return _TURN.get()


def tracked(rel: str, exclude: "Iterable[str]") -> bool:
    return not rel.startswith(_SKIP_PREFIXES) and rel not in exclude


def _key(snap_value) -> tuple:
    # (mtime_ns, size, digest) as `_snapshot_workdir_files` records it; the digest is None off FAT-like volumes.
    return (snap_value[0], snap_value[1], snap_value[2] if len(snap_value) > 2 else None)


def _probe(workdir: str, rels: "Iterable[str]") -> dict:
    """Regular files among *rels*, keyed as a walk keys them; one behind a linked folder counts as gone."""
    root = os.path.realpath(workdir)
    found = {}
    for rel in rels:
        path = os.path.join(root, *rel.split("/"))
        try:
            if os.path.realpath(os.path.dirname(path)) != os.path.dirname(path):
                continue
            info = os.lstat(path)
        except OSError:
            continue
        if stat_mod.S_ISREG(info.st_mode):
            found[rel] = (info.st_mtime_ns, info.st_size, None)
    return found


class FileHistory:
    """One per account. Every method is blocking: call it from a worker thread."""

    def __init__(self, root: str):
        self.root = root
        self.objects = os.path.join(root, "objects")
        os.makedirs(os.path.join(self.objects, "tmp"), exist_ok = True)
        self._db = sqlite3.connect(
            os.path.join(root, "index.sqlite"), check_same_thread = False, isolation_level = None
        )
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(_SCHEMA)
        self._lock = threading.RLock()
        self._state: dict = {}  # workdir -> rel -> _Entry, write-through cache of workdir_state
        self._active: dict = {}  # workdir -> [CallToken]
        self._restoring: set = set()
        self._since_gc = 0
        self._added = 0  # bytes stored since the last gc

    # Blobs

    def _blob_path(self, digest: str) -> str:
        return os.path.join(self.objects, digest[:2], digest[2:])

    def _put_blob(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        path = self._blob_path(digest)
        try:
            stored = os.path.getsize(path)
        except OSError:
            stored = None
        if stored is not None:
            # The row too: one lost to a crash or a rolled-back call would leave the file uncounted.
            self._db.execute(
                "INSERT OR IGNORE INTO blobs VALUES(?,?,?,?)",
                (digest, len(data), stored, time.time()),
            )
            return digest
        probe = data[: 64 << 10]
        if _ZSTD_C is not None:
            codec, compress = _ZSTD, _ZSTD_C.compress
        else:
            codec, compress = _ZLIB, lambda chunk: zlib.compress(chunk, 1)
        # Weights and archives do not compress: a probe keeps them from paying for the attempt.
        if probe and len(compress(probe)) < 0.9 * len(probe):
            payload = codec + compress(data)
        else:
            payload = _RAW + data
        os.makedirs(os.path.dirname(path), exist_ok = True)
        tmp = os.path.join(self.objects, "tmp", uuid.uuid4().hex)
        with open(tmp, "wb") as fh:
            fh.write(payload)
        os.replace(tmp, path)
        self._db.execute(
            "INSERT OR IGNORE INTO blobs VALUES(?,?,?,?)",
            (digest, len(data), len(payload), time.time()),
        )
        self._added += len(payload)
        return digest

    def _get_blob(self, digest: str) -> bytes:
        with open(self._blob_path(digest), "rb") as fh:
            payload = fh.read()
        codec, body = payload[:1], payload[1:]
        try:
            if codec == _ZLIB:
                data = zlib.decompress(body)
            elif codec == _ZSTD:
                if _ZSTD_D is None:
                    raise OSError("this file was stored with zstd, which is not installed")
                data = _ZSTD_D.decompress(body)
            else:
                data = body
        except OSError:
            raise
        except Exception as exc:  # zlib.error, zstd.ZstdError: one file's error, not the rewind's
            raise OSError("stored copy is damaged") from exc
        # A torn write after a crash is reported, never restored.
        if hashlib.sha256(data).hexdigest() != digest:
            raise OSError("stored copy is damaged")
        return data

    # Known state

    def _load_state(self, workdir: str) -> dict:
        state = self._state.get(workdir)
        if state is None:
            state = {
                row[0]: _Entry(*row[1:])
                for row in self._db.execute(
                    "SELECT rel_path, mtime_ns, size, digest, kind, blob, mode FROM workdir_state WHERE workdir=?",
                    (workdir,),
                )
            }
            self._state[workdir] = state
        return state

    def _capture(self, workdir: str, rel: str, key: tuple, budget: Optional[list]) -> _Entry:
        mtime_ns, size, digest = key
        data, mode = None, None
        if size <= MAX_FILE_BYTES and (budget is None or budget[0] >= size):
            try:
                fd = os.open(
                    os.path.join(workdir, rel),
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0),
                )
                with os.fdopen(fd, "rb") as fh:
                    mode = stat_mod.S_IMODE(os.fstat(fh.fileno()).st_mode)
                    data = fh.read(MAX_FILE_BYTES + 1)
            except OSError:
                data = None
        if data is None or len(data) > MAX_FILE_BYTES:
            return _Entry(mtime_ns, size, digest, "untracked", None, mode)
        if budget is not None:
            budget[0] -= len(data)
        return _Entry(mtime_ns, size, digest, "blob", self._put_blob(data), mode)

    def _sync(
        self,
        workdir: str,
        snap: dict,
        paths: Optional[set] = None,
        complete: bool = True,
    ) -> dict:
        """Bring the known state up to ``snap``: rel -> (old, new), either side None when absent.
        ``paths`` limits the sync to those names (edit_file knows its one file); ``complete`` says
        whether a name missing from ``snap`` is really gone (a capped walk cannot say so)."""
        state = self._load_state(workdir)
        first = (
            not state
            and self._db.execute("SELECT 1 FROM workdirs WHERE workdir=?", (workdir,)).fetchone()
            is None
        )
        # Per sync, not only the baseline: one call writing thousands of large files must not copy them all.
        budget = [BASELINE_BUDGET_BYTES if first else CALL_BUDGET_BYTES]
        changes: dict = {}
        names = snap.keys() if first or paths is None else paths
        for rel in names:
            if rel not in snap:
                continue
            key = _key(snap[rel])
            old = state.get(rel)
            if old is not None and (old.mtime_ns, old.size, old.digest) == key:
                continue
            new = self._capture(workdir, rel, key, budget)
            if old is not None and old.kind == new.kind == "blob" and old.blob == new.blob:
                # Same content, so a known digest still holds; a probe that hashed nothing must not drop it.
                old.mtime_ns, old.mode = new.mtime_ns, new.mode
                old.digest = old.digest if new.digest is None else new.digest
                self._db.execute(
                    "UPDATE workdir_state SET mtime_ns=?, digest=?, mode=? WHERE workdir=? AND rel_path=?",
                    (old.mtime_ns, old.digest, old.mode, workdir, rel),
                )
                continue
            changes[rel] = (old, new)
        gone = (paths or set()) if paths is not None else (state.keys() if complete else ())
        for rel in [r for r in gone if r in state and r not in snap]:
            changes[rel] = (state[rel], None)
        for rel, (_old, new) in changes.items():
            if new is None:
                state.pop(rel, None)
                self._db.execute(
                    "DELETE FROM workdir_state WHERE workdir=? AND rel_path=?", (workdir, rel)
                )
            else:
                state[rel] = new
                self._db.execute(
                    "INSERT OR REPLACE INTO workdir_state VALUES(?,?,?,?,?,?,?,?)",
                    (
                        workdir,
                        rel,
                        new.mtime_ns,
                        new.size,
                        new.digest,
                        new.kind,
                        new.blob,
                        new.mode,
                    ),
                )
        if first:
            self._db.execute("INSERT OR IGNORE INTO workdirs VALUES(?,?)", (workdir, time.time()))
            return {}  # the baseline is where history starts, not a change
        return changes

    def _walk_sync(self, workdir: str, walk) -> dict:
        snap, complete = walk(workdir)
        changes = self._sync(workdir, snap, None, complete)
        if not complete:
            # A capped walk never reached these: read them directly, or a rewind trusts a stale state
            # and overwrites the user's later edit.
            missed = {rel for rel in self._load_state(workdir) if rel not in snap}
            changes.update(self._sync(workdir, _probe(workdir, missed), missed))
        return changes

    def _record(
        self,
        chat_id,
        workdir,
        turn_id,
        kind,
        changes,
        ambiguous = False,
    ) -> Optional[int]:
        if not changes:
            return None
        cur = self._db.execute(
            "INSERT INTO checkpoints(chat_id, workdir, turn_id, kind, ambiguous, created_at) VALUES(?,?,?,?,?,?)",
            (chat_id, workdir, turn_id, kind, int(ambiguous), time.time()),
        )
        rows = []
        for rel, (old, new) in changes.items():
            rows.append(
                (
                    cur.lastrowid,
                    rel,
                    "absent" if old is None else old.kind,
                    old.blob if old else None,
                    old.mode if old else None,
                    "absent" if new is None else new.kind,
                    new.blob if new else None,
                    new.mode if new else None,
                )
            )
        self._db.executemany("INSERT INTO file_versions VALUES(?,?,?,?,?,?,?,?)", rows)
        return cur.lastrowid

    def _transaction(self, workdir: str, body):
        self._db.execute("BEGIN")
        try:
            result = body()
            self._db.execute("COMMIT")
            return result
        except BaseException:
            self._db.execute("ROLLBACK")
            self._state.pop(workdir, None)  # rebuilt from the index on next use
            raise

    def needs_baseline(self, workdir: str) -> bool:
        workdir = os.path.realpath(workdir)
        with self._lock:
            return (
                not self._load_state(workdir)
                and self._db.execute(
                    "SELECT 1 FROM workdirs WHERE workdir=?", (workdir,)
                ).fetchone()
                is None
            )

    # Hooks around a tool call

    def begin(
        self,
        chat_id,
        workdir,
        turn_id,
        snap,
        paths = None,
        complete = True,
    ) -> CallToken:
        workdir = os.path.realpath(workdir)
        with self._lock:
            if workdir in self._restoring:
                raise RewindBusy("a rewind is running in this folder")
            token = CallToken(chat_id, workdir, turn_id)
            running = self._active.setdefault(workdir, [])

            def body():
                # Kept even when the call changes nothing: deleting the chat releases its folder's images.
                self._db.execute(
                    "INSERT OR IGNORE INTO chat_workdirs VALUES(?,?)", (chat_id, workdir)
                )
                changes = self._sync(workdir, snap, paths, complete)
                if running:
                    # Another chat's call is writing this folder right now: its writes cannot be told
                    # apart from ours, so they go to the call that started first and both are flagged.
                    owner = running[0]
                    owner.ambiguous = token.ambiguous = True
                    self._record(owner.chat_id, workdir, owner.turn_id, "tool", changes, True)
                else:
                    self._record(None, workdir, None, "external", changes)

            self._transaction(workdir, body)
            running.append(token)
            return token

    def end(
        self,
        token: CallToken,
        snap,
        paths = None,
        complete = True,
    ) -> Optional[int]:
        if token.done:
            return None
        token.done = True
        with self._lock:
            try:
                checkpoint = self._transaction(
                    token.workdir,
                    lambda: self._record(
                        token.chat_id,
                        token.workdir,
                        token.turn_id,
                        "tool",
                        self._sync(token.workdir, snap, paths, complete),
                        token.ambiguous,
                    ),
                )
            finally:
                running = self._active.get(token.workdir, [])
                if token in running:
                    running.remove(token)
                if not running:
                    self._active.pop(token.workdir, None)
            if checkpoint is not None:
                self._trim_chat(token.chat_id)
            return checkpoint

    # Rewind

    def rewind(
        self,
        chat_id,
        turn_ids,
        walk,
        *,
        dry_run = False,
        force = False,
        busy = None,
    ) -> RewindReport:
        """Put the files ``chat_id`` changed during ``turn_ids`` back to how they were before the
        first of those turns touched them. ``walk(workdir)`` returns a fresh ``(snapshot, complete)``. Only files
        this chat changed are touched; one changed since by anything else is skipped as a conflict."""
        report = RewindReport()
        turn_ids = [t for t in dict.fromkeys(turn_ids) if t]
        if not turn_ids:
            return report
        with self._lock:
            marks = ",".join("?" * len(turn_ids))
            rows = self._db.execute(
                "SELECT c.workdir, fv.rel_path, fv.before_kind, fv.before_blob, fv.before_mode, c.ambiguous"
                " FROM file_versions fv"
                f" JOIN checkpoints c ON c.id=fv.checkpoint_id WHERE c.chat_id=? AND c.turn_id IN ({marks})"
                " ORDER BY c.id",
                (chat_id, *turn_ids),
            ).fetchall()
            if not rows:
                return report
            report.has_changes = True
            workdirs = sorted({row[0] for row in rows})
            for workdir in workdirs:
                if (
                    self._active.get(workdir)
                    or workdir in self._restoring
                    or (busy and busy(workdir))
                ):
                    raise RewindBusy("a tool call is running in this chat's folder")
            if not dry_run:
                self._restoring.update(workdirs)
        undo_turn = f"rewind:{uuid.uuid4().hex}"
        try:
            for workdir in workdirs:
                self._rewind_folder(chat_id, workdir, rows, walk, report, dry_run, force, undo_turn)
        finally:
            if not dry_run:
                with self._lock:
                    self._restoring.difference_update(workdirs)
        if not dry_run and (report.restored or report.deleted or report.recreated):
            report.undo_turn_id = undo_turn
        return report

    def _rewind_folder(self, chat_id, workdir, rows, walk, report, dry_run, force, undo_turn):
        with self._lock:
            # Edits made since the last call (by the user, or another chat) are stored first, so the
            # rewind can tell them apart from this chat's own and they stay restorable too.
            self._transaction(
                workdir,
                lambda: self._record(
                    None, workdir, None, "external", self._walk_sync(workdir, walk)
                ),
            )
            target: dict = {}
            # Written while another chat's call ran in this folder: either chat's, so never ours to undo.
            shared = set()
            for wd, rel, kind, blob, mode, ambiguous in rows:
                if wd == workdir:
                    target.setdefault(rel, (kind, blob, mode))
                    if ambiguous:
                        shared.add(rel)
            last_after = {
                row[0]: (row[1], row[2])
                for row in self._db.execute(
                    "SELECT fv.rel_path, fv.after_kind, fv.after_blob FROM file_versions fv"
                    " JOIN checkpoints c ON c.id=fv.checkpoint_id WHERE c.chat_id=? AND c.workdir=? ORDER BY c.id",
                    (chat_id, workdir),
                )
            }
            state = dict(self._load_state(workdir))
        for rel in sorted(target):
            kind, blob, mode = target[rel]
            current = state.get(rel)
            now = ("absent", None) if current is None else (current.kind, current.blob)
            if kind == "untracked" or now[0] == "untracked":
                report.untracked.append(rel)
                continue
            if now == (kind, blob):
                continue
            if (now != last_after.get(rel) or rel in shared) and not force:
                report.conflicts.append(rel)
                continue
            bucket = (
                report.deleted
                if kind == "absent"
                else report.restored
                if current
                else report.recreated
            )
            if dry_run:
                bucket.append(rel)
                continue
            try:
                if kind == "absent":
                    _safe_unlink(workdir, rel)
                else:
                    _safe_write(workdir, rel, self._get_blob(blob), mode)
                bucket.append(rel)
            except OSError as exc:
                report.errors.append(f"{rel}: {exc}")
        if dry_run:
            return
        with self._lock:
            self._transaction(
                workdir,
                lambda: self._record(
                    chat_id, workdir, undo_turn, "rewind", self._walk_sync(workdir, walk)
                ),
            )

    # Retention

    def _trim_chat(self, chat_id) -> None:
        with self._lock:
            count = self._db.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE chat_id=?", (chat_id,)
            ).fetchone()[0]
            if count > CHAT_CAP_CHECKPOINTS:
                self._db.execute(
                    "DELETE FROM checkpoints WHERE id IN"
                    " (SELECT id FROM checkpoints WHERE chat_id=? ORDER BY id LIMIT ?)",
                    (chat_id, count - CHAT_CAP_CHECKPOINTS),
                )
            self._since_gc += 1
            # By count, or sooner by bytes: a few calls rewriting large files can pass the cap alone.
            if self._since_gc >= 50 or self._added >= GC_AFTER_BYTES:
                self._since_gc = self._added = 0
                self.gc()

    def delete_chat(self, chat_id) -> None:
        with self._lock:
            workdirs = [
                row[0]
                for row in self._db.execute(
                    "SELECT workdir FROM chat_workdirs WHERE chat_id=?"
                    " UNION SELECT workdir FROM checkpoints WHERE chat_id=?",
                    (chat_id, chat_id),
                )
            ]
            self._db.execute("DELETE FROM checkpoints WHERE chat_id=?", (chat_id,))
            self._db.execute("DELETE FROM chat_workdirs WHERE chat_id=?", (chat_id,))
            # A folder no other chat uses keeps no stored images either.
            for workdir in workdirs:
                if (
                    workdir not in self._active
                    and self._db.execute(
                        "SELECT 1 FROM chat_workdirs WHERE workdir=? UNION ALL"
                        " SELECT 1 FROM checkpoints WHERE workdir=? AND chat_id IS NOT NULL LIMIT 1",
                        (workdir, workdir),
                    ).fetchone()
                    is None
                ):
                    self._forget_workdir(workdir)
            self.gc()

    def _forget_workdir(self, workdir: str) -> None:
        self._db.execute("DELETE FROM checkpoints WHERE workdir=?", (workdir,))
        self._db.execute("DELETE FROM workdir_state WHERE workdir=?", (workdir,))
        self._db.execute("DELETE FROM workdirs WHERE workdir=?", (workdir,))
        self._state.pop(workdir, None)

    def gc(self) -> int:
        """Oldest checkpoints go first while the store is over its cap, then unreferenced blobs."""
        with self._lock:
            self._db.execute(
                "DELETE FROM checkpoints WHERE kind='external' AND id NOT IN (SELECT checkpoint_id FROM file_versions)"
            )
            # Counts history only: images of the files folders hold now cannot be freed by dropping
            # checkpoints, and counting them would drop every checkpoint once they pass the cap.
            while (
                self._db.execute(
                    "SELECT COALESCE(SUM(stored),0) FROM blobs b"
                    " WHERE NOT EXISTS (SELECT 1 FROM workdir_state WHERE blob=b.hash)"
                ).fetchone()[0]
                > GLOBAL_CAP_BYTES
            ):
                oldest = self._db.execute("SELECT MIN(id) FROM checkpoints").fetchone()[0]
                if oldest is None:
                    break
                self._db.execute("DELETE FROM checkpoints WHERE id=?", (oldest,))
                self._sweep()
            return self._sweep()

    def _sweep(self) -> int:
        dead = [
            row[0]
            for row in self._db.execute(
                "SELECT hash FROM blobs b WHERE NOT EXISTS (SELECT 1 FROM workdir_state WHERE blob=b.hash)"
                " AND NOT EXISTS (SELECT 1 FROM file_versions WHERE before_blob=b.hash)"
                " AND NOT EXISTS (SELECT 1 FROM file_versions WHERE after_blob=b.hash)"
            )
        ]
        self._db.executemany("DELETE FROM blobs WHERE hash=?", [(digest,) for digest in dead])
        # After the rows: a crash leaves an orphan file, never a row pointing at nothing.
        for digest in dead:
            try:
                os.unlink(self._blob_path(digest))
            except FileNotFoundError:
                pass
        return len(dead)


def _resolve_inside(workdir: str, rel: str, create_dirs: bool) -> str:
    parts = rel.split("/")
    if os.path.isabs(rel) or any(part in ("", ".", "..") for part in parts):
        raise OSError("not a path inside the chat's folder")
    # The folder itself too: recorded by its real path, so a link there now leads somewhere else.
    _require_plain_dir(workdir, "the chat's folder")
    current = workdir
    for part in parts[:-1]:
        current = os.path.join(current, part)
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if not create_dirs:
                raise
            os.mkdir(current)
            continue
        # A parent swapped for a link would carry the write outside the folder.
        _require_plain_dir(current, f"'{part}'", info)
    return os.path.join(current, parts[-1])


def _require_plain_dir(
    path: str,
    name: str,
    info = None,
) -> None:
    info = os.lstat(path) if info is None else info
    if not stat_mod.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise OSError(f"{name} is no longer a plain folder")


def _safe_write(workdir: str, rel: str, data: bytes, mode: Optional[int]) -> None:
    path = _resolve_inside(workdir, rel, create_dirs = True)
    tmp = os.path.join(os.path.dirname(path), f".unsloth-rewind-{uuid.uuid4().hex}.tmp")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    fd = os.open(tmp, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        if mode is not None:
            os.chmod(tmp, mode)
        if os.path.isfile(path):
            from core.library import replace_file
            replace_file(tmp, path)  # carries the Library's name and folder over to the new inode
        else:
            os.replace(tmp, path)  # replaces a link at the name, never writes through it
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _safe_unlink(workdir: str, rel: str) -> None:
    path = _resolve_inside(workdir, rel, create_dirs = False)
    if not stat_mod.S_ISREG(os.lstat(path).st_mode):
        raise OSError("no longer a regular file")
    os.unlink(path)


_stores: dict = {}
_stores_lock = threading.Lock()


def close_store(root) -> None:
    """Drop and close an account's history before its folders are moved, since an open
    index blocks the rename on Windows."""
    root = str(root)
    with _stores_lock:
        found = _stores.pop(root, None)
    if found is not None:
        with found._lock:
            found._db.close()


def store() -> FileHistory:
    """The current account's history."""
    from utils.paths.storage_roots import account_path

    root = str(account_path("file_history"))
    with _stores_lock:
        found = _stores.get(root)
        if found is None:
            found = _stores[root] = FileHistory(root)
        return found


def begin_call(
    session_id,
    workdir,
    snap,
    paths = None,
    complete = True,
) -> Optional[CallToken]:
    """Start tracking a tool call of the current turn. None, and nothing done, outside a chat turn."""
    turn = _TURN.get()
    if turn is None or not session_id or not workdir:
        return None
    try:
        return store().begin(turn[0], workdir, turn[1], snap, paths, complete)
    except Exception:
        logger.warning("file history: could not start tracking a tool call", exc_info = True)
        return None


def end_call(
    token: Optional[CallToken],
    snap,
    paths = None,
    complete = True,
) -> None:
    if token is None:
        return
    try:
        store().end(token, snap, paths, complete)
    except Exception:
        logger.warning("file history: could not record a tool call", exc_info = True)
