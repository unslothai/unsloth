# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama.cpp converter sources (convert_lora_to_gguf.py, gguf-py) from the Unsloth fork's releases.

Each unslothai/llama.cpp release carries ``llama.cpp-source-<tag>.tar.gz`` and the
``llama-prebuilt-sha256.json`` that pins its digest, so no git and no ggml-org download is
needed. The backend cannot import the studio/ installer scripts (see
utils/prebuilt/runtime_libs.py), hence this small standalone copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Optional

FORK_REPO = "unslothai/llama.cpp"
SHA256_ASSET_NAME = "llama-prebuilt-sha256.json"
_USER_AGENT = "unsloth-studio-llama-cpp-source"


def source_asset_name(tag: str) -> str:
    return f"llama.cpp-source-{tag}.tar.gz"


def is_fork_release_tag(repo: Optional[str], tag: Optional[str]) -> bool:
    return repo == FORK_REPO and bool(tag) and "-mix-" in tag


def _asset_url(tag: Optional[str], name: str) -> str:
    if tag is None:
        return f"https://github.com/{FORK_REPO}/releases/latest/download/{name}"
    return (
        f"https://github.com/{FORK_REPO}/releases/download/"
        f"{urllib.parse.quote(tag, safe = '')}/{urllib.parse.quote(name, safe = '')}"
    )


def _open(url: str, timeout: int = 60):
    headers = {"User-Agent": _USER_AGENT}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and urllib.parse.urlparse(url).hostname == "api.github.com":
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers = headers), timeout = timeout)


def _fetch_json(url: str):
    with _open(url) as response:
        return json.loads(response.read().decode("utf-8"))


def _matching_fork_tag(upstream_tag: str) -> Optional[str]:
    """Newest non-draft fork release built on ``upstream_tag`` (tags are ``<upstream>-mix-<sha>``)."""
    prefix = f"{upstream_tag}-mix-"
    for page in range(1, 21):
        releases = _fetch_json(
            f"https://api.github.com/repos/{FORK_REPO}/releases?per_page=100&page={page}"
        )
        if not isinstance(releases, list) or not releases:
            return None
        for release in releases:
            tag = release.get("tag_name") or ""
            if (
                not release.get("draft")
                and not release.get("prerelease")
                and tag.startswith(prefix)
            ):
                return tag
    return None


def resolve_fork_release(repo: Optional[str], tag: Optional[str]) -> tuple[str, dict]:
    """(fork release tag, its llama-prebuilt-sha256.json) for a converter revision.

    A fork tag is used as is; a bare tag the fork reports is tried as a release first. Anything
    else (an old ggml-org marker, nothing) maps to the fork release built on that upstream tag,
    else the latest.
    """
    fork_tag: Optional[str] = None
    checksums = None
    if tag and ("-mix-" in tag or repo == FORK_REPO):
        try:
            checksums = _fetch_json(_asset_url(tag, SHA256_ASSET_NAME))
            fork_tag = tag
        except Exception:
            if "-mix-" in tag:
                raise
    if checksums is None:
        if tag:
            # A failed lookup raises: falling back to latest would drop the installed revision.
            fork_tag = _matching_fork_tag(tag.split("-mix-")[0])
        checksums = _fetch_json(_asset_url(fork_tag, SHA256_ASSET_NAME))
    if not isinstance(checksums, dict):
        raise RuntimeError(f"{SHA256_ASSET_NAME} from {FORK_REPO} is not a JSON object")
    release_tag = checksums.get("release_tag") or fork_tag
    if not isinstance(release_tag, str) or not release_tag:
        raise RuntimeError(f"{SHA256_ASSET_NAME} from {FORK_REPO} names no release_tag")
    if fork_tag is not None and release_tag != fork_tag:
        raise RuntimeError(
            f"{SHA256_ASSET_NAME} for {FORK_REPO}@{fork_tag} describes release {release_tag}"
        )
    return release_tag, checksums


def source_artifact(checksums: dict, release_tag: str) -> tuple[str, str]:
    """(asset name, sha256) of the release's source archive: the tag-named one, else the
    exact-commit one (``llama.cpp-source-commit-<source_commit>.tar.gz``) the installer accepts."""
    artifacts = checksums.get("artifacts") or {}
    names = [source_asset_name(release_tag)]
    commit = checksums.get("source_commit")
    if isinstance(commit, str) and re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        names.append(f"llama.cpp-source-commit-{commit.lower()}.tar.gz")
    for name in names:
        entry = artifacts.get(name)
        digest = entry.get("sha256") if isinstance(entry, dict) else None
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            return name, digest.lower()
    raise RuntimeError(f"{FORK_REPO}@{release_tag} publishes no sha256 for its source archive")


def safe_extract_tar(archive: Path, destination: Path) -> None:
    """Extract a .tar.gz refusing absolute paths, '..', links leaving ``destination``, devices."""
    base = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            name = member.name.replace("\\", "/")
            parts = PurePosixPath(name).parts
            if name.startswith("/") or ".." in parts or re.match(r"^[A-Za-z]:", name):
                raise RuntimeError(f"unsafe path in llama.cpp source archive: {member.name}")
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise RuntimeError(f"unsupported entry in llama.cpp source archive: {member.name}")
            if member.issym() or member.islnk():
                link = member.linkname.replace("\\", "/")
                origin = (base / name).parent if member.issym() else base
                if link.startswith("/") or not (origin / link).resolve().is_relative_to(base):
                    raise RuntimeError(f"link escapes the llama.cpp source archive: {member.name}")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(destination, members = members, filter = "data")
        else:  # Python < 3.11.4; every member was checked above.
            tar.extractall(destination, members = members)


def download_converter_source(repo: Optional[str], tag: Optional[str], parent: Path) -> Path:
    """Download, verify and publish ``parent/llama.cpp-source-<fork tag>``; returns that dir."""
    release_tag, checksums = resolve_fork_release(repo, tag)
    target = Path(parent) / f"llama.cpp-source-{release_tag}"
    if (target / "convert_lora_to_gguf.py").is_file():
        return target
    asset, expected = source_artifact(checksums, release_tag)
    Path(parent).mkdir(parents = True, exist_ok = True)
    with tempfile.TemporaryDirectory(dir = parent, prefix = ".llama.cpp-source-") as tmp:
        archive = Path(tmp) / asset
        digest = hashlib.sha256()
        with (
            _open(_asset_url(release_tag, archive.name), timeout = 300) as response,
            open(archive, "wb") as out,
        ):
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError(
                f"{archive.name} sha256 mismatch: expected {expected}, got {digest.hexdigest()}"
            )
        extract_dir = Path(tmp) / "extract"
        extract_dir.mkdir()
        safe_extract_tar(archive, extract_dir)
        roots = [p for p in extract_dir.iterdir() if p.is_dir()]
        if len(roots) != 1 or not (roots[0] / "convert_lora_to_gguf.py").is_file():
            raise RuntimeError(
                f"{archive.name} has no convert_lora_to_gguf.py under one top directory"
            )
        if not target.exists():
            try:
                os.replace(roots[0], target)
            except OSError:
                if not (target / "convert_lora_to_gguf.py").is_file():
                    raise
    return target
