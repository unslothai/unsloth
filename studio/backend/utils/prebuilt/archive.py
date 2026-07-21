# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Traversal/symlink-guarded zip + tar.gz extraction for the prebuilt installers."""

from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
from pathlib import Path

from .errors import PrebuiltFallback


def _safe_extract_path(base: Path, member_name: str) -> Path:
    member_path = Path(member_name.replace("\\", "/"))
    if member_path.is_absolute():
        raise PrebuiltFallback(f"archive member used an absolute path: {member_name}")
    target = (base / member_path).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError as exc:
        raise PrebuiltFallback(f"archive member escaped destination: {member_name}") from exc
    return target


def _extract_zip_safely(source: Path, base: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            target = _safe_extract_path(base, member.filename)
            mode = (member.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise PrebuiltFallback(f"zip archive contained a symlink entry: {member.filename}")
            if member.is_dir():
                target.mkdir(parents = True, exist_ok = True)
                continue
            target.parent.mkdir(parents = True, exist_ok = True)
            with archive.open(member, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            perm = (member.external_attr >> 16) & 0o777
            if perm & 0o111:
                os.chmod(target, target.stat().st_mode | 0o111)


def _extract_tar_safely(source: Path, base: Path) -> None:
    # Bundles co-locate the server with its libs; some ship versioned symlinks
    # (libwhisper.so -> libwhisper.so.1), so defer links until the files exist.
    pending_links: list[tuple[tarfile.TarInfo, Path]] = []
    with tarfile.open(source, "r:gz") as archive:
        for member in archive.getmembers():
            target = _safe_extract_path(base, member.name)
            if member.isdir():
                target.mkdir(parents = True, exist_ok = True)
                continue
            if member.islnk() or member.issym():
                pending_links.append((member, target))
                continue
            if not member.isfile():
                raise PrebuiltFallback(f"tar archive contained an unsupported entry: {member.name}")
            target.parent.mkdir(parents = True, exist_ok = True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise PrebuiltFallback(f"tar archive entry could not be read: {member.name}")
            with extracted, target.open("wb") as dst:
                shutil.copyfileobj(extracted, dst)
            if member.mode & 0o111:
                os.chmod(target, target.stat().st_mode | 0o111)

    for member, target in pending_links:
        link_name = member.linkname.replace("\\", "/")
        link_path = Path(link_name)
        if link_path.is_absolute() or not link_name:
            raise PrebuiltFallback(
                f"archive link used an unsafe target: {member.name} -> {link_name}"
            )
        # tar symlink names are link-parent relative; hard-link names are archive-root relative.
        resolved = (target.parent / link_path if member.issym() else base / link_path).resolve()
        try:
            resolved.relative_to(base.resolve())
        except ValueError as exc:
            raise PrebuiltFallback(
                f"archive link escaped destination: {member.name} -> {link_name}"
            ) from exc
        target.parent.mkdir(parents = True, exist_ok = True)
        if target.exists() or target.is_symlink():
            target.unlink()
        if member.issym():
            target.symlink_to(link_name)
        else:  # hard link
            shutil.copy2(resolved, target)


def extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents = True, exist_ok = True)
    if archive_path.name.endswith(".zip"):
        _extract_zip_safely(archive_path, destination)
    elif archive_path.name.endswith(".tar.gz"):
        _extract_tar_safely(archive_path, destination)
    else:
        raise PrebuiltFallback(f"unsupported archive format: {archive_path.name}")
