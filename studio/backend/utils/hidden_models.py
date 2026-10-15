# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Infra-only model detection shared by the model routes and the hub inventory. Lives directly under ``utils`` (not ``utils.models``) so the hub cache scanner can import it without pulling in ``utils/models/__init__.py``, which eagerly loads the model-config/checkpoint stack, and without importing ``routes.models`` (import-time side effects, would cycle)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")

_PROBE_REPO_ID = "ggml-org/models"
# Include ".gguf" so repos like user/stories260K-finetune-GGUF are not hidden.
_PROBE_FILENAME = "stories260k.gguf"
_DEFAULT_EMBEDDING_REPO_IDS = {
    "unsloth/bge-small-en-v1.5",
    "unsloth/bge-small-en-v1.5-GGUF",
}
_DEFAULT_EMBEDDING_PATH_BASENAMES = {"bge-small-en-v1.5"}
# STT checkpoints; GGUF companions have no config.json, so they must be listed by id.
_HIDDEN_STT_REPO_IDS = frozenset(
    {
        "unsloth/whisper-tiny",
        "unsloth/whisper-base",
        "unsloth/whisper-small",
        "unsloth/whisper-large-v3-turbo",
        "unsloth/whisper-large-v3",
        "unslothai/whisper-tiny-GGUF",
        "unslothai/whisper-base-GGUF",
        "unslothai/whisper-small-GGUF",
        "unslothai/whisper-large-v3-turbo-GGUF",
        "unslothai/whisper-large-v3-GGUF",
        "unslothai/Qwen3-ASR-0.6B-GGUF",
        "unslothai/Qwen3-ASR-1.7B-GGUF",
    }
)
_HIDDEN_STT_REPO_IDS_LOWER = frozenset(repo_id.lower() for repo_id in _HIDDEN_STT_REPO_IDS)

# audio.cpp umbrella repo; its GGUFs are reached through the audio.cpp catalog.
_HIDDEN_AUDIO_CPP_REPO_IDS_LOWER = frozenset({"audio-cpp/audio.cpp-gguf"})

# TTS checkpoints stay visible but not chat-loadable; config sniffing cannot catch them.
_CURATED_TTS_REPO_IDS = frozenset(
    {
        "unsloth/orpheus-3b-0.1-ft",
        "unsloth/orpheus-3b-0.1-ft-bnb-4bit",
        "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit",
        "unsloth/orpheus-3b-0.1-ft-GGUF",
        "canopylabs/orpheus-3b-0.1-ft",
        "unsloth/csm-1b",
        "sesame/csm-1b",
        "unsloth/Spark-TTS-0.5B",
        "unsloth/Llama-OuteTTS-1.0-1B",
    }
)
_CURATED_TTS_REPO_IDS_LOWER = frozenset(repo_id.lower() for repo_id in _CURATED_TTS_REPO_IDS)


def is_curated_tts_repo_id(value: str | None) -> bool:
    """True only for Unsloth's exact curated TTS Hub repositories."""
    return bool(value and value.strip().lower() in _CURATED_TTS_REPO_IDS_LOWER)


def is_audio_cpp_repo_id(value: str | None) -> bool:
    """True for audio.cpp's umbrella repo. Hidden from chat like a curated STT repo, and like one still
    emitted as a Model Hub management row, since that row is where its downloaded packages are deleted."""
    return bool(value and value.strip().lower() in _HIDDEN_AUDIO_CPP_REPO_IDS_LOWER)


def is_curated_stt_repo_id(value: str | None) -> bool:
    """True only for Unsloth's exact curated STT Hub repositories. Still hidden from chat, but task-scoped inventory consumers need the real cache rows so the Audio page need not reimplement size, format, variants and lifecycle."""
    return bool(value and value.strip().lower() in _HIDDEN_STT_REPO_IDS_LOWER)


def _config_is_whisper(path: Path) -> bool:
    """True if a config.json declares a Whisper model."""
    try:
        with open(path, "r", encoding = "utf-8") as file:
            config = json.load(file)
    except Exception:
        return False
    if not isinstance(config, dict):
        return False
    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type.strip().lower() == "whisper":
        return True
    architectures = config.get("architectures")
    return isinstance(architectures, list) and any(
        isinstance(name, str) and name == "WhisperForConditionalGeneration"
        for name in architectures
    )


def _path_is_whisper_model(value: str) -> bool:
    """Inspect an existing local model path's config; never hides name-only matches."""
    if _HF_REPO_ID_RE.fullmatch(value.strip()):
        return False
    path = Path(value).expanduser()
    try:
        if path.is_file():
            path = path.parent
        candidates = [path / "config.json"]
        snapshots = path / "snapshots"
        if snapshots.is_dir():
            candidates.extend(child / "config.json" for child in snapshots.iterdir())
    except OSError:
        return False
    return any(_config_is_whisper(candidate) for candidate in candidates)


def _safe_resolve(path: Path) -> Optional[str]:
    """resolve() to a string, or None when the path is inaccessible."""
    try:
        return str(path.resolve())
    except OSError:
        return None


def _existing_resolved_path(value: str) -> Optional[str]:
    """Resolve an existing local path."""
    path = Path(value).expanduser()
    try:
        if not path.exists():
            return None
    except OSError:
        return None
    return _safe_resolve(path)


def _path_contains_repo_id(value: str, repo_ids: set[str]) -> bool:
    """Match exact repo-derived path segments."""
    parts = [part for part in value.lower().replace("\\", "/").split("/") if part]
    for repo_id in repo_ids:
        owner, name = repo_id.split("/", 1)
        if f"models--{owner}--{name}" in parts:
            return True
        if any(
            parts[index] == owner and parts[index + 1] == name for index in range(len(parts) - 1)
        ):
            return True
    return False


def _path_basename_is_default_embedder(value: str) -> bool:
    """Match a default embedder folder or a suffixed local weight filename."""
    normalized = value.lower().replace("\\", "/").rstrip("/")
    basename = normalized.rsplit("/", 1)[-1]
    return any(
        basename == needle
        or any(basename.startswith(f"{needle}{separator}") for separator in ("-", "_", "."))
        for needle in _DEFAULT_EMBEDDING_PATH_BASENAMES
    )


def is_hidden_model(*values: str | None) -> bool:
    """Repo ids match exactly, not by substring, so a generic embedder basename cannot hide other repos."""
    from core.rag import config as rag_config

    hidden_repo_ids = {
        _PROBE_REPO_ID.lower(),
        *(repo_id.lower() for repo_id in _DEFAULT_EMBEDDING_REPO_IDS),
        *_HIDDEN_STT_REPO_IDS_LOWER,
        *_HIDDEN_AUDIO_CPP_REPO_IDS_LOWER,
    }
    exact_paths: list[str] = []
    for model in {
        rag_config.EMBEDDING_MODEL,
        rag_config.default_gguf_repo(),
        rag_config.effective_embedding_model(),
        rag_config.effective_gguf_repo(),
    }:
        existing_path = _existing_resolved_path(model)
        if existing_path:
            exact_paths.append(existing_path.lower())
        elif _HF_REPO_ID_RE.match(model):
            hidden_repo_ids.add(model.lower())
        else:
            resolved = _safe_resolve(Path(model).expanduser())
            if resolved:
                exact_paths.append(resolved.lower())
    for v in values:
        if not v:
            continue
        low = v.lower()
        if _HF_REPO_ID_RE.match(v):
            if low in hidden_repo_ids:
                return True
            continue
        # Split on both separators so Windows and POSIX paths match on either OS.
        if low.replace("\\", "/").rsplit("/", 1)[-1] == _PROBE_FILENAME:
            return True
        if _path_basename_is_default_embedder(v):
            return True
        if _path_contains_repo_id(v, hidden_repo_ids):
            return True
        if _path_is_whisper_model(v):
            return True
        if exact_paths:
            resolved = _safe_resolve(Path(v).expanduser())
            if resolved and resolved.lower() in exact_paths:
                return True
    return False
