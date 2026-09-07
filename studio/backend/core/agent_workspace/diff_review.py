# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Structured, bounded Git diff manifests for local review surfaces."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

from . import common as workspace_common
from .common import AgentWorkspaceError
from .git_service import (
    _git,
    _literal_pathspecs,
    _project_git,
    _safe_git_root,
    _workspace_writer_slot,
    workspace_fingerprint,
)
from .worktrees import owned_worktree_path


_DIFF_MODES = frozenset({"head", "staged", "unstaged"})
_CONFLICT_CODES = frozenset({"DD", "AU", "UD", "UA", "DU", "AA", "UU"})
_HUNK_HEADER = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_lines>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_lines>\d+))? @@(?P<label>.*)$"
)
_RAW_HEADER = re.compile(
    r"^:(?P<old_mode>[0-7]{6}) (?P<new_mode>[0-7]{6}) "
    r"(?P<old_sha>[0-9a-f]+) (?P<new_sha>[0-9a-f]+) "
    r"(?P<status>[A-Z][0-9]{0,3})$"
)

DEFAULT_MAX_BYTES = 512_000
MAX_MAX_BYTES = 2_000_000
MAX_FILES = 5_000
MAX_HUNKS = 20_000
MAX_LINES = 200_000
MAX_LINE_CHARS = 64_000
_MANIFEST_VERSION = 1


def _mode_arguments(mode: str) -> list[str]:
    if mode == "head":
        return ["HEAD"]
    if mode == "staged":
        return ["--cached"]
    if mode == "unstaged":
        return []
    raise AgentWorkspaceError("Diff review mode must be head, staged, or unstaged.")


def _diff_pathspecs(pathspec: str) -> list[str]:
    if pathspec == ".":
        return []
    return _literal_pathspecs([pathspec])


def _target(project_id: str, worktree_id: Optional[str]) -> tuple[Path, Path, str, dict]:
    if worktree_id is None:
        root, repository = _project_git(project_id)
        relative = root.relative_to(repository).as_posix()
        return (
            root,
            repository,
            "." if relative == "." else relative,
            {"kind": "primary", "worktreeId": None},
        )
    normalized = str(worktree_id).strip()
    if not normalized or len(normalized) > 128 or any(char in normalized for char in "\x00\r\n"):
        raise AgentWorkspaceError("Studio worktree ID is invalid.")
    root = owned_worktree_path(project_id, normalized)
    repository = _safe_git_root(root)
    if root.resolve(strict = True) != repository:
        raise AgentWorkspaceError("Studio worktree root no longer matches its Git checkout.")
    return root, repository, ".", {"kind": "worktree", "worktreeId": normalized}


def _status_snapshot(repository: Path, pathspec: str) -> tuple[list[str], list[dict], bool]:
    output, truncated = _git(
        repository,
        [
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            *_diff_pathspecs(pathspec),
        ],
        output_limit = 1_000_000,
        timeout_seconds = 20,
        neutralize_filters = True,
    )
    if truncated or "\ufffd" in output:
        return [], [], True
    records = [record for record in output.split("\0") if record]
    conflicts: list[str] = []
    untracked: list[dict] = []
    index = 0
    while index < len(records):
        record = records[index]
        if len(record) < 3 or record[2] != " ":
            return [], [], True
        code = record[:2]
        path = record[3:]
        if code in _CONFLICT_CODES:
            conflicts.append(path)
        if code == "??":
            untracked.append({"code": code, "path": path, "oldPath": None})
        if code[0] in {"R", "C"} or code[1] in {"R", "C"}:
            index += 1
            if index >= len(records):
                return [], [], True
        index += 1
    return (
        conflicts[:MAX_FILES],
        untracked[:MAX_FILES],
        len(conflicts) > MAX_FILES or len(untracked) > MAX_FILES,
    )


def _raw_entries(output: str) -> list[dict]:
    tokens = output.split("\0")
    if tokens and tokens[-1] == "":
        tokens.pop()
    entries: list[dict] = []
    index = 0
    while index < len(tokens):
        match = _RAW_HEADER.fullmatch(tokens[index])
        if match is None or index + 1 >= len(tokens):
            raise AgentWorkspaceError("Git returned an invalid raw diff manifest.")
        status = match.group("status")
        first_path = tokens[index + 1]
        index += 2
        old_path = None
        path = first_path
        if status.startswith(("R", "C")):
            if index >= len(tokens):
                raise AgentWorkspaceError("Git returned an invalid rename diff manifest.")
            old_path = first_path
            path = tokens[index]
            index += 1
        if not path or "\x00" in path or (old_path is not None and not old_path):
            raise AgentWorkspaceError("Git returned an invalid diff path.")
        entries.append(
            {
                "code": status,
                "path": path,
                "oldPath": old_path,
                "oldMode": match.group("old_mode"),
                "newMode": match.group("new_mode"),
                "oldBlob": match.group("old_sha"),
                "newBlob": match.group("new_sha"),
            }
        )
        if len(entries) > MAX_FILES:
            raise OverflowError("file-limit")
    return entries


def _patch_sections(output: str) -> list[str]:
    if not output:
        return []
    marker = "diff --git "
    starts = [match.start() for match in re.finditer(r"(?m)^diff --git ", output)]
    if not starts or starts[0] != 0:
        raise AgentWorkspaceError("Git returned an invalid patch manifest.")
    return [
        output[start : (starts[index + 1] if index + 1 < len(starts) else len(output))]
        for index, start in enumerate(starts)
        if output[start:].startswith(marker)
    ]


def _structured_hunks(section: str) -> tuple[list[dict], int, int, int]:
    raw_lines = section.splitlines()
    hunks: list[dict] = []
    additions = 0
    deletions = 0
    line_count = 0
    index = 0
    while index < len(raw_lines):
        header_match = _HUNK_HEADER.match(raw_lines[index])
        if header_match is None:
            index += 1
            continue
        header = raw_lines[index]
        old_line = int(header_match.group("old_start"))
        new_line = int(header_match.group("new_start"))
        expected_old = int(header_match.group("old_lines") or "1")
        expected_new = int(header_match.group("new_lines") or "1")
        old_seen = 0
        new_seen = 0
        canonical_lines: list[str] = []
        lines: list[dict[str, Any]] = []
        index += 1
        while index < len(raw_lines) and not raw_lines[index].startswith("@@ "):
            raw = raw_lines[index]
            if raw.startswith("diff --git "):
                break
            if raw == r"\ No newline at end of file":
                if not lines:
                    raise AgentWorkspaceError("Git returned an invalid no-newline marker.")
                lines[-1]["noNewline"] = True
                canonical_lines.append(raw)
                index += 1
                continue
            if not raw or raw[0] not in {" ", "+", "-"}:
                break
            text = raw[1:]
            if len(text) > MAX_LINE_CHARS:
                raise OverflowError("line-size-limit")
            if raw[0] == " ":
                lines.append(
                    {"kind": "context", "text": text, "oldLine": old_line, "newLine": new_line}
                )
                old_line += 1
                new_line += 1
                old_seen += 1
                new_seen += 1
            elif raw[0] == "+":
                lines.append({"kind": "add", "text": text, "oldLine": None, "newLine": new_line})
                additions += 1
                new_line += 1
                new_seen += 1
            else:
                lines.append({"kind": "delete", "text": text, "oldLine": old_line, "newLine": None})
                deletions += 1
                old_line += 1
                old_seen += 1
            canonical_lines.append(raw)
            line_count += 1
            if line_count > MAX_LINES:
                raise OverflowError("line-limit")
            index += 1
        if old_seen != expected_old or new_seen != expected_new:
            raise AgentWorkspaceError("Git returned an internally inconsistent diff hunk.")
        hunks.append(
            {
                "header": header,
                "oldStart": int(header_match.group("old_start")),
                "oldLines": expected_old,
                "newStart": int(header_match.group("new_start")),
                "newLines": expected_new,
                "lines": lines,
                "_canonicalLines": canonical_lines,
            }
        )
        if len(hunks) > MAX_HUNKS:
            raise OverflowError("hunk-limit")
    return hunks, additions, deletions, line_count


def _stable_id(kind: str, payload: dict) -> str:
    encoded = json.dumps(
        {"version": _MANIFEST_VERSION, "kind": kind, **payload},
        ensure_ascii = False,
        separators = (",", ":"),
        sort_keys = True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_manifest(
    entry: dict, section: str, *, mode: str, head: str, fingerprint: str
) -> tuple[dict, int, int, int]:
    hunks, additions, deletions, line_count = _structured_hunks(section)
    binary = "\nGIT binary patch\n" in f"\n{section}" or "\nBinary files " in f"\n{section}"
    code_kind = str(entry["code"])[0]
    mode_changed = (
        entry["oldMode"] != "000000"
        and entry["newMode"] != "000000"
        and entry["oldMode"] != entry["newMode"]
    )
    whole_file_only = binary or code_kind in {"R", "C", "T", "U"} or mode_changed or not hunks
    identity = {
        "mode": mode,
        "head": head,
        "sourceFingerprint": fingerprint,
        "code": entry["code"],
        "path": entry["path"],
        "oldPath": entry["oldPath"],
        "oldMode": entry["oldMode"],
        "newMode": entry["newMode"],
        "oldBlob": entry["oldBlob"],
        "newBlob": entry["newBlob"],
    }
    public_hunks = []
    for hunk in hunks:
        canonical_lines = hunk.pop("_canonicalLines")
        hunk["id"] = _stable_id(
            "hunk",
            {**identity, "header": hunk["header"], "lines": canonical_lines},
        )
        public_hunks.append(hunk)
    return (
        {
            "selectionId": _stable_id("file", identity),
            "code": entry["code"],
            "path": entry["path"],
            "oldPath": entry["oldPath"],
            "oldMode": entry["oldMode"],
            "newMode": entry["newMode"],
            "binary": binary,
            "wholeFileOnly": whole_file_only,
            "additions": additions,
            "deletions": deletions,
            "hunks": [] if whole_file_only else public_hunks,
        },
        0 if whole_file_only else len(public_hunks),
        len(public_hunks),
        line_count,
    )


def _untracked_manifest(item: dict, *, mode: str, head: str, fingerprint: str) -> dict:
    identity = {
        "mode": mode,
        "head": head,
        "sourceFingerprint": fingerprint,
        "code": "??",
        "path": item["path"],
        "oldPath": None,
    }
    return {
        "selectionId": _stable_id("file", identity),
        "code": "??",
        "path": item["path"],
        "oldPath": None,
        "oldMode": "000000",
        "newMode": "000000",
        "binary": False,
        "wholeFileOnly": True,
        "additions": 0,
        "deletions": 0,
        "hunks": [],
    }


def _blocked_manifest(
    *,
    project_id: str,
    target: dict,
    mode: str,
    head: str,
    fingerprint: str,
    reasons: list[str],
    conflicts: Optional[list[str]] = None,
    truncated: bool = False,
    max_bytes: int,
) -> dict:
    return {
        "version": _MANIFEST_VERSION,
        "projectId": project_id,
        "target": target,
        "mode": mode,
        "head": head,
        "sourceFingerprint": fingerprint,
        "selectable": False,
        "blockedReasons": reasons,
        "conflictedPaths": conflicts or [],
        "files": [],
        "fileCount": 0,
        "hunkCount": 0,
        "lineCount": 0,
        "truncated": truncated,
        "limits": {
            "maxBytes": max_bytes,
            "maxFiles": MAX_FILES,
            "maxHunks": MAX_HUNKS,
            "maxLines": MAX_LINES,
            "maxLineChars": MAX_LINE_CHARS,
        },
    }


def build_diff_manifest(
    project_id: str,
    *,
    mode: str = "head",
    worktree_id: Optional[str] = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict:
    """Return a coherent structured diff whose selections are bound to repository state."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in _DIFF_MODES:
        raise AgentWorkspaceError("Diff review mode must be head, staged, or unstaged.")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise AgentWorkspaceError("Diff review byte limit is invalid.")
    bounded_bytes = max(4_096, min(max_bytes, MAX_MAX_BYTES))
    root, repository, pathspec, target = _target(project_id, worktree_id)
    with _workspace_writer_slot(root):
        head, _ = _git(
            repository,
            ["rev-parse", "--verify", "HEAD^{commit}"],
            output_limit = 256,
            timeout_seconds = 5,
        )
        head = head.strip()
        fingerprint_before = workspace_fingerprint(root)
        if not workspace_common.workspace_fingerprint_complete(fingerprint_before):
            return _blocked_manifest(
                project_id = project_id,
                target = target,
                mode = normalized_mode,
                head = head,
                fingerprint = fingerprint_before,
                reasons = ["incomplete-source-fingerprint"],
                max_bytes = bounded_bytes,
            )

        conflicts, untracked, status_incomplete = _status_snapshot(repository, pathspec)
        if status_incomplete:
            return _blocked_manifest(
                project_id = project_id,
                target = target,
                mode = normalized_mode,
                head = head,
                fingerprint = fingerprint_before,
                reasons = ["status-truncated-or-invalid"],
                truncated = True,
                max_bytes = bounded_bytes,
            )
        if conflicts:
            return _blocked_manifest(
                project_id = project_id,
                target = target,
                mode = normalized_mode,
                head = head,
                fingerprint = fingerprint_before,
                reasons = ["repository-conflicts"],
                conflicts = conflicts,
                max_bytes = bounded_bytes,
            )

        mode_args = _mode_arguments(normalized_mode)
        raw, raw_truncated = _git(
            repository,
            [
                "diff",
                *mode_args,
                "--raw",
                "-z",
                "--no-abbrev",
                "--find-renames",
                "--no-ext-diff",
                "--no-textconv",
                "--",
                *_diff_pathspecs(pathspec),
            ],
            output_limit = min(1_000_000, bounded_bytes),
            timeout_seconds = 20,
            neutralize_filters = True,
        )
        patch, patch_truncated = _git(
            repository,
            [
                "diff",
                *mode_args,
                "--patch",
                "--binary",
                "--full-index",
                "--find-renames",
                "--no-ext-diff",
                "--no-textconv",
                "--no-color",
                "--unified=3",
                "--",
                *_diff_pathspecs(pathspec),
            ],
            output_limit = bounded_bytes,
            timeout_seconds = 20,
            neutralize_filters = True,
        )
        final_head, _ = _git(
            repository,
            ["rev-parse", "--verify", "HEAD^{commit}"],
            output_limit = 256,
            timeout_seconds = 5,
        )
        fingerprint_after = workspace_fingerprint(root)

    if raw_truncated or patch_truncated or "\ufffd" in raw or "\ufffd" in patch:
        return _blocked_manifest(
            project_id = project_id,
            target = target,
            mode = normalized_mode,
            head = head,
            fingerprint = fingerprint_before,
            reasons = ["diff-truncated-or-undecodable"],
            truncated = True,
            max_bytes = bounded_bytes,
        )
    if final_head.strip() != head or fingerprint_after != fingerprint_before:
        return _blocked_manifest(
            project_id = project_id,
            target = target,
            mode = normalized_mode,
            head = head,
            fingerprint = fingerprint_before,
            reasons = ["workspace-changed-during-review"],
            max_bytes = bounded_bytes,
        )

    try:
        if raw.count("\0") > MAX_FILES * 3:
            raise OverflowError("file-limit")
        if patch and patch.count("\n") + 1 > MAX_LINES:
            raise OverflowError("line-limit")
        entries = _raw_entries(raw)
        sections = _patch_sections(patch)
        if len(entries) != len(sections):
            raise AgentWorkspaceError("Git diff metadata did not match its patch output.")
        files = []
        hunk_count = 0
        parsed_hunk_count = 0
        line_count = 0
        for entry, section in zip(entries, sections):
            file, file_hunks, parsed_file_hunks, file_lines = _file_manifest(
                entry,
                section,
                mode = normalized_mode,
                head = head,
                fingerprint = fingerprint_before,
            )
            hunk_count += file_hunks
            parsed_hunk_count += parsed_file_hunks
            line_count += file_lines
            if parsed_hunk_count > MAX_HUNKS:
                raise OverflowError("hunk-limit")
            if line_count > MAX_LINES:
                raise OverflowError("line-limit")
            files.append(file)
        if normalized_mode in {"head", "unstaged"}:
            seen = {(item["code"], item["path"]) for item in files}
            for item in untracked:
                if ("??", item["path"]) not in seen:
                    files.append(
                        _untracked_manifest(
                            item,
                            mode = normalized_mode,
                            head = head,
                            fingerprint = fingerprint_before,
                        )
                    )
        if len(files) > MAX_FILES:
            raise OverflowError("file-limit")
    except (AgentWorkspaceError, OverflowError) as exc:
        reason = str(exc) if isinstance(exc, OverflowError) else "diff-parse-invalid"
        return _blocked_manifest(
            project_id = project_id,
            target = target,
            mode = normalized_mode,
            head = head,
            fingerprint = fingerprint_before,
            reasons = [reason],
            truncated = isinstance(exc, OverflowError),
            max_bytes = bounded_bytes,
        )

    return {
        "version": _MANIFEST_VERSION,
        "projectId": project_id,
        "target": target,
        "mode": normalized_mode,
        "head": head,
        "sourceFingerprint": fingerprint_before,
        "selectable": True,
        "blockedReasons": [],
        "conflictedPaths": [],
        "files": files,
        "fileCount": len(files),
        "hunkCount": hunk_count,
        "lineCount": line_count,
        "truncated": False,
        "limits": {
            "maxBytes": bounded_bytes,
            "maxFiles": MAX_FILES,
            "maxHunks": MAX_HUNKS,
            "maxLines": MAX_LINES,
            "maxLineChars": MAX_LINE_CHARS,
        },
    }


__all__ = ["build_diff_manifest"]
