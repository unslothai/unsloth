# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Build the Studio notebook catalog from model_defaults YAML and notebooks_index.json."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from utils.models.model_config import MODEL_NAME_MAPPING

NOTEBOOKS_REPO = "unslothai/notebooks"
NOTEBOOKS_NB_DIR = "nb"

_BASED_ON_RE = re.compile(
    r"#\s*Based on\s+(.+?\.ipynb|.+?\.py)",
    re.IGNORECASE,
)
_MODEL_DEFAULTS_FOR_RE = re.compile(
    r"#\s*Model defaults for\s+(\S+)",
    re.IGNORECASE,
)


def _assets_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "assets"


def _defaults_dir() -> Path:
    return _assets_dir() / "configs" / "model_defaults"


def _index_path() -> Path:
    return _assets_dir() / "configs" / "notebooks_index.json"


def notebook_colab_url(notebook_file: str) -> str:
    return (
        f"https://colab.research.google.com/github/{NOTEBOOKS_REPO}/blob/main/"
        f"{NOTEBOOKS_NB_DIR}/{notebook_file}"
    )


def notebook_github_url(notebook_file: str) -> str:
    return f"https://github.com/{NOTEBOOKS_REPO}/blob/main/" f"{NOTEBOOKS_NB_DIR}/{notebook_file}"


def _notebook_id(notebook_file: str) -> str:
    stem = notebook_file.removesuffix(".ipynb").removesuffix(".py")
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")


def _human_title(notebook_file: str) -> str:
    stem = notebook_file.removesuffix(".ipynb").removesuffix(".py")
    return stem.replace("_", " ")


def _normalize_search(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _infer_category(notebook_file: str) -> str:
    name = notebook_file.lower()
    if "grpo" in name or "gspo" in name:
        return "grpo"
    if any(token in name for token in ("vision", "_vl_", "ocr", "pixtral", "deepseek")):
        return "vision"
    if any(
        token in name
        for token in ("tts", "whisper", "orpheus", "llasa", "csm", "spark_tts", "oute")
    ):
        return "audio"
    if any(token in name for token in ("embedding", "minilm", "bge", "modernbert")):
        return "embedding"
    if any(token in name for token in ("inference", "deployment", "phone")):
        return "inference"
    if any(token in name for token in ("thinking", "reasoning", "codeforces")):
        return "reasoning"
    if any(token in name for token in ("tool_calling", "coder", "codegemma", "code")):
        return "code"
    if "raft" in name:
        return "raft"
    if "classification" in name or "bert" in name:
        return "classification"
    return "sft"


def _is_public_notebook_file(name: str) -> bool:
    if not name.endswith(".ipynb"):
        return False
    if name.startswith("Kaggle-"):
        return False
    if name.startswith("AMD-"):
        return False
    return True


def _studio_model_for_yaml(yaml_filename: str, header_lines: list[str]) -> str | None:
    mapped = MODEL_NAME_MAPPING.get(yaml_filename)
    if mapped:
        return mapped[0]
    for line in header_lines[:5]:
        match = _MODEL_DEFAULTS_FOR_RE.match(line.strip())
        if match:
            return match.group(1)
    return None


def _extract_notebook_files(header_text: str) -> list[str]:
    files: list[str] = []
    for match in _BASED_ON_RE.finditer(header_text):
        raw = match.group(1).strip()
        for part in re.split(r"\s+and\s+", raw, flags = re.IGNORECASE):
            part = part.strip().strip('"').strip("'")
            if not part:
                continue
            if part.endswith(".ipynb"):
                files.append(part)
            elif part.endswith(".py"):
                files.append(part.replace(".py", ".ipynb"))
            else:
                files.append(f"{part}.ipynb")
    return files


def _make_entry(
    notebook_file: str,
    *,
    title: str | None = None,
    category: str | None = None,
    featured: bool | None = None,
    studio_model: str | None = None,
    entry_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": entry_id or _notebook_id(notebook_file),
        "title": title or _human_title(notebook_file),
        "notebook_file": notebook_file,
        "category": category or _infer_category(notebook_file),
        "featured": bool(featured),
        "studio_model": studio_model,
        "colab_url": notebook_colab_url(notebook_file),
        "github_url": notebook_github_url(notebook_file),
    }


def _merge_entries(
    existing: dict[str, Any] | None, incoming: dict[str, Any] | None
) -> dict[str, Any]:
    if not existing:
        return dict(incoming or {})
    if not incoming:
        return existing
    merged = {**existing, **incoming}
    if existing.get("studio_model") and not incoming.get("studio_model"):
        merged["studio_model"] = existing["studio_model"]
    notebook_file = incoming.get("notebook_file") or existing.get("notebook_file")
    if (
        existing.get("title")
        and isinstance(notebook_file, str)
        and incoming.get("title") == _human_title(notebook_file)
    ):
        merged["title"] = existing["title"]
    merged["featured"] = bool(existing.get("featured")) or bool(incoming.get("featured"))
    return merged


def _load_notebooks_index() -> tuple[list[str], frozenset[str], dict[str, dict[str, Any]]]:
    path = _index_path()
    if not path.is_file():
        return [], frozenset(), {}
    with open(path, encoding = "utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return [], frozenset(), {}

    raw_files = data.get("notebook_files")
    files = sorted(
        {
            name
            for name in (raw_files if isinstance(raw_files, list) else [])
            if isinstance(name, str) and _is_public_notebook_file(name)
        },
    )

    raw_featured = data.get("featured")
    featured = frozenset(
        name
        for name in (raw_featured if isinstance(raw_featured, list) else [])
        if isinstance(name, str)
    )

    raw_overrides = data.get("overrides")
    overrides: dict[str, dict[str, Any]] = {}
    if isinstance(raw_overrides, dict):
        for notebook_file, fields in raw_overrides.items():
            if isinstance(notebook_file, str) and isinstance(fields, dict):
                overrides[notebook_file] = fields

    return files, featured, overrides


def _normalize_override(
    notebook_file: str, fields: dict[str, Any], *, featured_names: frozenset[str]
) -> dict[str, Any]:
    title = fields.get("title")
    if not isinstance(title, str) or not title.strip():
        title = None
    category = fields.get("category")
    if not isinstance(category, str) or not category.strip():
        category = None
    studio_model = fields.get("studio_model")
    if studio_model is not None and not isinstance(studio_model, str):
        studio_model = None
    featured = bool(fields.get("featured")) or notebook_file in featured_names
    return _make_entry(
        notebook_file,
        title = title,
        category = category,
        featured = featured,
        studio_model = studio_model,
    )


def _build_enriched_map(
    *, featured_names: frozenset[str], overrides: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_file: dict[str, dict[str, Any]] = {}

    defaults_dir = _defaults_dir()
    if defaults_dir.is_dir():
        for config_path in sorted(defaults_dir.rglob("*.yaml")):
            if config_path.name == "default.yaml":
                continue
            yaml_filename = config_path.name
            with open(config_path, encoding = "utf-8") as handle:
                header_lines = [handle.readline() for _ in range(12)]
            header_text = "".join(header_lines)
            notebook_files = _extract_notebook_files(header_text)
            if not notebook_files:
                continue
            studio_model = _studio_model_for_yaml(yaml_filename, header_lines)
            for notebook_file in notebook_files:
                if not _is_public_notebook_file(notebook_file):
                    continue
                entry = _make_entry(
                    notebook_file,
                    studio_model = studio_model,
                    featured = notebook_file in featured_names,
                )
                by_file[notebook_file] = _merge_entries(by_file.get(notebook_file), entry)

    for notebook_file, fields in overrides.items():
        if not _is_public_notebook_file(notebook_file):
            continue
        entry = _normalize_override(
            notebook_file,
            fields,
            featured_names = featured_names,
        )
        by_file[notebook_file] = _merge_entries(by_file.get(notebook_file), entry)

    return by_file


def notebook_matches_query(entry: dict[str, Any], query: str | None) -> bool:
    normalized_query = _normalize_search(query or "")
    if not normalized_query:
        return True
    haystack = "".join(
        [
            _normalize_search(str(entry.get("title", ""))),
            _normalize_search(str(entry.get("notebook_file", ""))),
            _normalize_search(str(entry.get("studio_model") or "")),
            _normalize_search(str(entry.get("category", ""))),
        ],
    )
    return normalized_query in haystack


@lru_cache(maxsize = 8)
def build_notebook_catalog(query: str | None = None) -> list[dict[str, Any]]:
    public_files, featured_names, overrides = _load_notebooks_index()
    enriched = _build_enriched_map(
        featured_names = featured_names,
        overrides = overrides,
    )

    by_file: dict[str, dict[str, Any]] = {}
    if public_files:
        for notebook_file in public_files:
            entry = _make_entry(
                notebook_file,
                featured = notebook_file in featured_names,
            )
            by_file[notebook_file] = _merge_entries(entry, enriched.get(notebook_file))
    else:
        by_file = dict(enriched)

    for notebook_file, entry in enriched.items():
        if notebook_file not in by_file:
            by_file[notebook_file] = entry

    catalog = list(by_file.values())
    if query:
        catalog = [entry for entry in catalog if notebook_matches_query(entry, query)]

    catalog.sort(
        key = lambda item: (
            not item.get("featured", False),
            item.get("category", ""),
            item.get("title", "").lower(),
        ),
    )
    return catalog
