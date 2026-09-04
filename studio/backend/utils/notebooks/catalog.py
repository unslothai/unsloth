# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Build the Studio notebook catalog from the upstream index and local metadata."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from utils.models.model_config import MODEL_NAME_MAPPING

_REPO = "unslothai/notebooks"
_NB_DIR = "nb"
NOTEBOOKS_INDEX_URL = f"https://api.github.com/repos/{_REPO}/contents/{_NB_DIR}?ref=main"

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "assets" / "configs"
_OVERRIDES_PATH = _CONFIG_DIR / "notebooks_index.json"
_DEFAULTS_DIR = _CONFIG_DIR / "model_defaults"

_FETCH_TIMEOUT = 5.0
_MAX_INDEX_BYTES = 2 * 1024 * 1024
_CACHE_TTL = 15 * 60
_FAILURE_TTL = 60
_OFFLINE_VALUES = {"1", "true", "yes", "on"}

_BASED_ON_RE = re.compile(r"#\s*Based on\s+(.+)", re.IGNORECASE)
_MODEL_RE = re.compile(r"#\s*Model defaults for\s+(\S+)", re.IGNORECASE)
_CATEGORY_RULES = (
    ("grpo", ("grpo", "gspo")),
    ("vision", ("vision", "_vl_", "ocr", "pixtral", "deepseek")),
    ("audio", ("tts", "whisper", "orpheus", "llasa", "csm", "spark_tts", "oute")),
    ("embedding", ("embedding", "minilm", "bge", "modernbert")),
    ("inference", ("inference", "deployment", "phone")),
    ("reasoning", ("thinking", "reasoning", "codeforces")),
    ("code", ("tool_calling", "coder", "codegemma", "code")),
    ("raft", ("raft",)),
    ("classification", ("classification", "bert")),
)

logger = logging.getLogger(__name__)
_cache_lock = threading.Lock()
_cached_files: tuple[str, ...] | None = None
_cache_until = 0.0


def _is_public(name: str) -> bool:
    return (
        name.endswith(".ipynb")
        and "/" not in name
        and "\\" not in name
        and not name.startswith(("AMD-", "Kaggle-"))
    )


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _category(name: str) -> str:
    lowered = name.lower()
    return next(
        (
            category
            for category, tokens in _CATEGORY_RULES
            if any(token in lowered for token in tokens)
        ),
        "sft",
    )


def _load_overrides() -> dict[str, dict[str, Any]]:
    if not _OVERRIDES_PATH.is_file():
        return {}
    with open(_OVERRIDES_PATH, encoding = "utf-8") as handle:
        data = json.load(handle)
    overrides = data.get("overrides", {}) if isinstance(data, dict) else {}
    return {
        name: fields
        for name, fields in overrides.items()
        if isinstance(name, str) and _is_public(name) and isinstance(fields, dict)
    }


def _load_studio_models() -> dict[str, str]:
    models: dict[str, str] = {}
    if not _DEFAULTS_DIR.is_dir():
        return models

    for path in sorted(_DEFAULTS_DIR.rglob("*.yaml")):
        if path.name == "default.yaml":
            continue
        with open(path, encoding = "utf-8") as handle:
            header = "".join(handle.readline() for _ in range(3))

        based_on = _BASED_ON_RE.search(header)
        mapped = MODEL_NAME_MAPPING.get(path.name)
        model_match = _MODEL_RE.search(header)
        model = mapped[0] if mapped else model_match.group(1) if model_match else None
        if not based_on or not model:
            continue

        for name in re.findall(
            r"(?:^|\s+and\s+)(.+?\.(?:ipynb|py))",
            based_on.group(1),
            re.IGNORECASE,
        ):
            notebook = re.sub(r"\.py$", ".ipynb", name.strip("\"'"), flags = re.IGNORECASE)
            if _is_public(notebook):
                models[notebook] = model
    return models


def _fetch_repo_notebook_files() -> list[str] | None:
    request = urllib.request.Request(
        NOTEBOOKS_INDEX_URL,
        headers = {
            "Accept": "application/vnd.github+json",
            "Accept-Encoding": "identity",
            "User-Agent": "unsloth-studio-notebooks",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout = _FETCH_TIMEOUT) as response:
            body = response.read(_MAX_INDEX_BYTES + 1)
        if len(body) > _MAX_INDEX_BYTES:
            return None
        payload = json.loads(body)
    except Exception as exc:
        logger.debug("Could not fetch notebook index: %s", exc)
        return None

    if not isinstance(payload, list):
        return None
    files = sorted(
        {
            item["name"]
            for item in payload
            if (
                isinstance(item, dict)
                and item.get("type") == "file"
                and isinstance(item.get("name"), str)
                and _is_public(item["name"])
            )
        }
    )
    return files or None


def _repo_notebook_files() -> list[str]:
    global _cached_files, _cache_until

    offline = any(
        os.environ.get(name, "").strip().lower() in _OFFLINE_VALUES
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )
    if offline:
        return list(_cached_files or ())

    with _cache_lock:
        if time.monotonic() < _cache_until:
            return list(_cached_files or ())

        files = _fetch_repo_notebook_files()
        if files is not None:
            _cached_files = tuple(files)
        _cache_until = time.monotonic() + (_CACHE_TTL if files is not None else _FAILURE_TTL)
        return list(_cached_files or ())


def _entry(
    notebook: str, overrides: dict[str, dict[str, Any]], studio_models: dict[str, str]
) -> dict[str, Any]:
    fields = overrides.get(notebook, {})
    stem = notebook.removesuffix(".ipynb")
    return {
        "id": re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-"),
        "title": _text(fields.get("title")) or stem.replace("_", " "),
        "notebook_file": notebook,
        "category": _text(fields.get("category")) or _category(notebook),
        "featured": bool(fields.get("featured")),
        "studio_model": _text(fields.get("studio_model")) or studio_models.get(notebook),
        "colab_url": (
            f"https://colab.research.google.com/github/{_REPO}/blob/main/{_NB_DIR}/{notebook}"
        ),
        "github_url": f"https://github.com/{_REPO}/blob/main/{_NB_DIR}/{notebook}",
    }


def notebook_matches_query(entry: dict[str, Any], query: str | None) -> bool:
    needle = re.sub(r"[^a-z0-9]+", "", (query or "").lower())
    haystack = "".join(
        str(entry.get(key) or "") for key in ("title", "notebook_file", "studio_model", "category")
    )
    return not needle or needle in re.sub(r"[^a-z0-9]+", "", haystack.lower())


def build_notebook_catalog(query: str | None = None) -> list[dict[str, Any]]:
    overrides = _load_overrides()
    studio_models = _load_studio_models()
    files = _repo_notebook_files() or sorted(overrides.keys() | studio_models.keys())
    catalog = [_entry(name, overrides, studio_models) for name in files]
    if query:
        catalog = [entry for entry in catalog if notebook_matches_query(entry, query)]
    return sorted(
        catalog,
        key = lambda entry: (
            not entry["featured"],
            entry["category"],
            entry["title"].lower(),
        ),
    )
