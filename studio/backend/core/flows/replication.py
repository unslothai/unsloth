# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export stamp in `meta.replication`: Studio version, per-model Hub revision and GGUF file, from the local cache only."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from core.flows.specs import SPECS_BY_TYPE

MAX_PINS = 32
_SHA = re.compile(r"^[0-9a-f]{40}$")
# Hub ids only: a local model path never leaves the machine in an export.
_HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _hub_cache() -> Optional[Path]:
    try:
        from utils.hf_cache_settings import active_hf_hub_cache
        return Path(active_hf_hub_cache())
    except Exception:  # noqa: BLE001 - outside a Studio process: the standard locations
        env = os.environ.get("HF_HUB_CACHE")
        if env:
            return Path(env)
        home = os.environ.get("HF_HOME")
        return Path(home) / "hub" if home else None


def _studio_version() -> Optional[str]:
    try:
        from utils.studio_version import get_studio_version
        return get_studio_version()
    except Exception:  # noqa: BLE001 - a version is a courtesy, never a reason to fail an export
        return None


def _choices(value: Any) -> Iterable[dict]:
    if isinstance(value, dict):
        if isinstance(value.get("repo"), str) and value["repo"].strip():
            yield value
            return
        for item in value.values():
            yield from _choices(item)
    elif isinstance(value, list):
        for item in value:
            yield from _choices(item)


def models_used(doc: dict) -> list[dict]:
    seen: dict[tuple[str, str], dict] = {}
    found = list((doc.get("requirements") or {}).get("models") or [])
    for node in doc.get("nodes") or []:
        found.extend(_choices(node.get("params") or {}))
    for choice in found:
        if not isinstance(choice, dict) or not isinstance(choice.get("repo"), str):
            continue
        repo = choice["repo"].strip()
        if not _HUB_ID.match(repo) or ".." in repo:
            continue
        variant = choice.get("variant") if isinstance(choice.get("variant"), str) else None
        if variant and (
            len(variant) > 200
            or variant.startswith("/")
            or ".." in variant
            or "\\" in variant
            or ":" in variant
        ):
            variant = None
        key = (repo.lower(), (variant or "").lower())
        if repo and key not in seen:
            seen[key] = {"repo": repo, "variant": variant}
    return list(seen.values())[:MAX_PINS]


def _repo_dir(cache: Path, repo: str) -> Path:
    return cache / ("models--" + repo.replace("/", "--"))


def cached_revision(cache: Optional[Path], repo: str) -> Optional[str]:
    if cache is None or "/" not in repo or ".." in repo:
        return None
    try:
        sha = (_repo_dir(cache, repo) / "refs" / "main").read_text(encoding = "utf-8").strip()
    except OSError:
        return None
    return sha if _SHA.match(sha) else None


def cached_gguf(
    cache: Optional[Path], repo: str, revision: Optional[str], variant: Optional[str]
) -> Optional[str]:
    if cache is None or not revision or not variant:
        return None
    root = _repo_dir(cache, repo) / "snapshots" / revision
    try:
        files = [p.relative_to(root).as_posix() for p in root.rglob("*.gguf")]
    except OSError:
        return None
    from core.inference.llama_cpp import _gguf_files_for_variant

    names = _gguf_files_for_variant(files, variant)
    return names[0] if names else None


def pins(
    doc: dict,
    cache: Optional[Path] = None,
    studio_version: Optional[str] = None,
) -> dict:
    cache = cache if cache is not None else _hub_cache()
    models = []
    for model in models_used(doc):
        entry: dict[str, Any] = {"repo": model["repo"]}
        if model["variant"]:
            entry["variant"] = model["variant"]
        revision = cached_revision(cache, model["repo"])
        if revision:
            entry["revision"] = revision
            gguf = cached_gguf(cache, model["repo"], revision, model["variant"])
            if gguf:
                entry["file"] = gguf
        models.append(entry)
    out: dict[str, Any] = {"models": models}
    version = studio_version if studio_version is not None else _studio_version()
    if version:
        out["studio_version"] = version
    return out


def stamp(
    doc: dict,
    cache: Optional[Path] = None,
    studio_version: Optional[str] = None,
) -> None:
    doc.setdefault("meta", {})["replication"] = pins(doc, cache, studio_version)


def notices(
    doc: dict,
    cache: Optional[Path] = None,
    studio_version: Optional[str] = None,
) -> list[str]:
    stamp_ = (doc.get("meta") or {}).get("replication")
    if not isinstance(stamp_, dict):
        return []
    cache = cache if cache is not None else _hub_cache()
    out: list[str] = []
    theirs = stamp_.get("studio_version")
    ours = studio_version if studio_version is not None else _studio_version()
    if isinstance(theirs, str) and ours and theirs != ours:
        out.append(f"Made with Unsloth Studio {theirs[:40]}; you have {ours[:40]}.")
    models = stamp_.get("models")
    for model in (models if isinstance(models, list) else [])[:MAX_PINS]:
        if not isinstance(model, dict) or not isinstance(model.get("repo"), str):
            continue
        made = model.get("revision")
        if not isinstance(made, str) or not _SHA.match(made):
            continue
        have = cached_revision(cache, model["repo"])
        name = model["repo"][:200]
        if have and have != made:
            out.append(
                f"Made with {name} revision {made[:12]}; you have {have[:12]}. Outputs may differ."
            )
        elif have and isinstance(model.get("file"), str):
            variant = model.get("variant") if isinstance(model.get("variant"), str) else None
            mine = cached_gguf(cache, model["repo"], have, variant)
            if mine and mine != model["file"]:
                out.append(f"Made with {name} file {model['file'][:200]}; you have {mine[:200]}.")
    return out


def live_conversations(doc: dict) -> bool:
    return any(
        (SPECS_BY_TYPE.get(n.get("type")) or {}).get("entrypoint")
        and (SPECS_BY_TYPE.get(n.get("type")) or {}).get("lane") == "gpu"
        and not n.get("disabled")
        for n in doc.get("nodes") or []
    )


LIVE_NOTE = (
    "Everything pre-generated plays without models; parts made while you play (characters' replies, "
    "new scenes) need a model."
)
