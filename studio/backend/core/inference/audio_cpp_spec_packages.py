# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The gguf packages in an audio.cpp model_specs entry: the repo each lives in, its files, the
primary GGUF the runtime is handed. Pure: no IO, no Studio imports."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

_REF_MAPS = ("files", "optional_files", "tensors", "optional_tensors")
# Licenses, notices, checksums and docs ship beside a GGUF but the runtime never reads them.
_PACKAGING_FILE = re.compile(
    r"(?i)(license|licence|notice|copying|readme|sha256sums|checksums)(\.[a-z0-9]+)?$|.*\.md$"
)


@dataclass(frozen = True)
class SpecPackage:
    id: str
    display_name: str
    precision: str
    repo: str
    target_directory: str
    # Repo paths of what the runtime reads; licenses and checksums the spec lists are left out.
    files: tuple[str, ...]
    strip_prefix: str
    default: bool
    recommended: bool
    # The runtime refuses a directory of several GGUFs, so it is handed this one file.
    primary: str

    @property
    def gguf_files(self) -> tuple[str, ...]:
        return tuple(f for f in self.files if f.lower().endswith(".gguf"))


def spec_packages(spec: Optional[dict]) -> tuple[SpecPackage, ...]:
    """The spec's gguf packages in spec order; a package with unsafe or misplaced files is dropped."""
    if not isinstance(spec, dict):
        return ()
    source = _gguf_source(spec)
    named = _named_files(source) if source is not None else set()
    defaults = _dict(_dict(spec.get("package_defaults")).get("download"))
    recommended = _dict(spec.get("ui")).get("recommended_package")
    packages = [
        _package(package, defaults, recommended, named)
        for package in _list(spec.get("packages"))
        if isinstance(package, dict) and _is_gguf(package.get("format"))
    ]
    return tuple(p for p in packages if p is not None)


def packages_for_repo(
    packages: Optional[Sequence[SpecPackage]],
    repo_id: str,
    umbrella_repo: str,
    folder: str = "",
) -> tuple[SpecPackage, ...]:
    """Packages in repo_id; a package without a repo of its own lives in umbrella_repo. With folder,
    only the packages in that folder of the umbrella."""
    repo = _text(repo_id).lower()
    if not repo:
        return ()
    umbrella = _text(umbrella_repo)
    folder = _text(folder).strip("/")
    return tuple(
        p
        for p in _packages(packages)
        if (p.repo or umbrella).lower() == repo and (not folder or _in_folder(p, folder))
    )


def available(
    packages: Optional[Sequence[SpecPackage]],
    present: Iterable[str],
    folder: str = "",
) -> tuple[SpecPackage, ...]:
    """Packages whose every file is in present (listed relative to folder when given), recommended first."""
    names = _names(present)
    folder = _text(folder).strip("/")
    prefix = folder + "/" if folder else ""
    return _ordered(
        tuple(p for p in _packages(packages) if all(_listed(f, prefix) in names for f in p.files))
    )


def variant_keys(packages: Optional[Sequence[SpecPackage]]) -> dict[str, str]:
    """Package id to variant key: its precision upper-cased when no other package here shares it, else its id."""
    items = _packages(packages)
    counts = Counter(p.precision for p in items)
    return {
        p.id: p.precision.upper() if p.precision and counts[p.precision] == 1 else p.id
        for p in items
    }


def _package(
    package: dict, defaults: dict, recommended: Any, named: set[str]
) -> Optional[SpecPackage]:
    package_id = _text(package.get("id"))
    raw = package.get("files")
    strip = package.get("strip_prefix", "")
    if not package_id or not isinstance(raw, list) or not raw:
        return None
    if not all(isinstance(f, str) for f in raw) or not isinstance(strip, str):
        return None
    files = tuple(f.replace("\\", "/") for f in raw)
    strip = strip.replace("\\", "/").strip("/")
    if not all(_safe_path(f) for f in files):
        return None
    files = tuple(f for f in files if not _PACKAGING_FILE.match(f.rsplit("/", 1)[-1]))
    if not files:
        return None
    if strip and not all(f.startswith(strip + "/") for f in files):
        return None

    gguf = [f for f in files if f.lower().endswith(".gguf")]
    # The GGUF no model: ref names (the "$gguf" root's own file); when every one is named or the
    # source has no "$gguf" root (LFM2.5, whose loader finds its parts by name), the first GGUF.
    candidates = [f for f in gguf if _local(strip, f) not in named]
    primary = (candidates or gguf or [""])[0]

    download = {**defaults, **_dict(package.get("download"))}
    return SpecPackage(
        id = package_id,
        display_name = _text(package.get("display_name")) or package_id,
        precision = _text(package.get("precision")).lower(),
        repo = _text(download.get("repo")).strip(),
        target_directory = _text(package.get("target_directory")),
        files = files,
        strip_prefix = strip,
        default = package.get("default") is True,
        recommended = package_id == recommended,
        primary = primary,
    )


def _gguf_source(spec: dict) -> Optional[dict]:
    for source in _list(spec.get("sources")):
        if isinstance(source, dict) and _is_gguf(source.get("format")):
            return source
    return None


def _named_files(source: dict) -> set[str]:
    """Relpaths the source's refs name beside its $gguf file: refs under any other root."""
    gguf_roots = {root for root, value in _dict(source.get("roots")).items() if value == "$gguf"}
    named = set()
    for key in _REF_MAPS:
        for value in _dict(source.get(key)).values():
            ref = value.get("source") if isinstance(value, dict) else value
            root, sep, rel = _text(ref).partition(":")
            if sep and rel and root not in gguf_roots:
                named.add(rel)
    return named


def _safe_path(path: str) -> bool:
    """No empty, '.', '..' or ':' part; an absolute path starts with an empty part, so it fails too."""
    return all(part not in ("", ".", "..") and ":" not in part for part in path.split("/"))


def _local(strip: str, repo_path: str) -> str:
    prefix = strip + "/"
    return repo_path[len(prefix) :] if strip and repo_path.startswith(prefix) else repo_path


def _listed(path: str, prefix: str) -> Optional[str]:
    """path as a listing of the folder shows it; None when it lies outside the folder."""
    if not prefix:
        return path
    return path[len(prefix) :] if path.startswith(prefix) else None


def _in_folder(package: SpecPackage, folder: str) -> bool:
    first = package.files[0].split("/", 1)[0] if package.files else ""
    return folder.lower() in (package.target_directory.lower(), first.lower())


def _ordered(packages: tuple[SpecPackage, ...]) -> tuple[SpecPackage, ...]:
    """Recommended first, then defaults, each keeping spec order."""
    return tuple(sorted(packages, key = lambda p: (not p.recommended, not p.default)))


def _packages(packages: Any) -> tuple[SpecPackage, ...]:
    if not isinstance(packages, (list, tuple)):
        return ()
    return tuple(p for p in packages if isinstance(p, SpecPackage))


def _names(values: Any) -> frozenset[str]:
    try:
        return frozenset(v for v in values if isinstance(v, str))
    except TypeError:
        return frozenset()


def _is_gguf(value: Any) -> bool:
    return isinstance(value, str) and value.lower() == "gguf"


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""
