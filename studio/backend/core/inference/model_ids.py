# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Public model identifiers for the OpenAI-compatible API.

The exposed API must report a stable, clean model id rather than the absolute
on-disk path of a local GGUF. The internal identifier for a direct local load is
the absolute ``.gguf`` path, which leaks the host filesystem layout and is
awkward for clients to round-trip. ``public_model_id`` maps such an internal
identifier to a clean name while leaving Hugging Face repo ids (``org/model``)
and already-clean names untouched.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

_GGUF_SUFFIX = ".gguf"


def _looks_like_path(identifier: str) -> bool:
    """A repo id has one slash: .gguf names, ./ or drive prefixes, and 3+ segments count as paths."""
    if identifier.lower().endswith(_GGUF_SUFFIX):
        return True
    # An audio.cpp umbrella id names a model folder of one Hub repo; three segments, not a path.
    if identifier.lower().startswith("audio-cpp/audio.cpp-gguf/"):
        from core.inference.audio_cpp_models import parse_identifier
        if parse_identifier(identifier) is not None:
            return False
    if identifier.startswith(("/", "\\", "./", "../", ".\\", "..\\", "~")):
        return True
    if len(identifier) >= 2 and identifier[1] == ":":
        return True
    if identifier.count("/") >= 2 or "\\" in identifier:
        return True
    return False


def hf_cache_repo_id(path: Optional[str]) -> Optional[str]:
    """Recovers org/name from a snapshot path, since the snapshot basename is only a commit hash."""
    if not path:
        return None
    parts = str(path).replace("\\", "/").split("/")
    for index, part in enumerate(parts):
        if part.startswith("models--") and parts[index + 1 : index + 2] == ["snapshots"]:
            return part[len("models--") :].replace("--", "/")
    return None


def public_model_id(identifier: Optional[str]) -> Optional[str]:
    """Returns a path-free public id: the repo id for HF cache paths, the stem for local GGUF files."""
    if not identifier:
        return identifier
    if not _looks_like_path(identifier):
        return identifier
    repo_id = hf_cache_repo_id(identifier)
    if repo_id:
        return repo_id
    name = os.path.basename(identifier.replace("\\", "/").rstrip("/"))
    if name.lower().endswith(_GGUF_SUFFIX):
        name = name[: -len(_GGUF_SUFFIX)]
    return name or identifier


def _is_hub_repo_id(identifier: str) -> bool:
    """``org/name``, including Hub repos named ``org/name.gguf``. A file reference
    carries a repo id plus a filename, so two or more slashes."""
    if identifier.count("/") != 1:
        return False
    stem = (
        identifier[: -len(_GGUF_SUFFIX)]
        if identifier.lower().endswith(_GGUF_SUFFIX)
        else identifier
    )
    return not _looks_like_path(stem)


def display_model_name(identifier: Optional[str]) -> Optional[str]:
    """Last segment of the public id; splitting the raw path would leak host layout on Windows."""
    if not identifier:
        return identifier
    if _is_hub_repo_id(identifier):
        return identifier.split("/")[1]
    clean = public_model_id(identifier)
    return clean.rsplit("/", 1)[-1] or clean


def model_id_matches(requested: Optional[str], internal: Optional[str]) -> bool:
    """Accepts the public id, and the raw internal id too for clients that cached a legacy path."""
    if requested is None or internal is None:
        return False
    if requested == internal:
        return True
    return public_model_id(internal) == requested


# Mirror Zoo’s MLX repository substitution without importing the ML stack.
_BNB_SUFFIXES = ("-unsloth-bnb-4bit", "-bnb-4bit")


def mlx_bnb_base_repo(model_name: Optional[str]) -> Optional[str]:
    """Return the replacement base repository, or None."""
    if not isinstance(model_name, str) or not model_name.startswith("unsloth/"):
        return None
    if os.path.exists(model_name):
        return None
    for suffix in _BNB_SUFFIXES:
        if model_name.endswith(suffix):
            return model_name[: -len(suffix)]
    return None


def mlx_host_bnb_base_repo(model_name: Optional[str]) -> Optional[str]:
    """Return the MLX replacement, excluding diffusion models."""
    import utils.hardware.hardware as hw
    from core.inference.diffusion_families import detect_family

    if hw.get_device() != hw.DeviceType.MLX:
        return None
    if not isinstance(model_name, str) or detect_family(model_name) is not None:
        return None
    return mlx_bnb_base_repo(model_name)


def mlx_bnb_substitutions(repos: Iterable[str]) -> list[tuple[str, str]]:
    swaps = []
    for repo in repos:
        base = mlx_bnb_base_repo(repo)
        if base:
            swaps.append((repo, base))
    return swaps


_DEEPSEEK_OCR_ARCHITECTURES = frozenset({"DeepseekOCRForCausalLM", "DeepseekOCR2ForCausalLM"})
# infer() tiles pages itself, so the 800 px chat cap blurs text; still bounded (many images per request).
DEEPSEEK_OCR_MAX_IMAGE_SIDE = 2048


def is_deepseek_ocr(model) -> bool:
    """A loaded DeepSeek-OCR model, which generates through its own infer()."""
    architectures = getattr(getattr(model, "config", None), "architectures", None) or ()
    return hasattr(model, "infer") and not _DEEPSEEK_OCR_ARCHITECTURES.isdisjoint(architectures)
