# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Q4NX folders converted on the Export page, listed and registered with FastFlowLM as local models.

FastFlowLM 1.0.12 only runs tags from its model_list.json (FLM_CONFIG_PATH overrides the file), and
reads a tag's files from FLM_MODEL_PATH/<model_path>/<name>. A converted folder is registered by
linking its files into such a folder next to the base model's config.json, under a tag whose entry
is the base's with xclbin_folder naming the base's kernels. The export folder is never written to.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from core.export.q4nx import SOURCE_FILE

LOCAL_PREFIX = "local-"
LOCAL_SIZE = "q4nx"
REQUIRED_FILES = ("model.q4nx", "tokenizer.json", "tokenizer_config.json")
OPTIONAL_FILES = ("chat_template.jinja",)
VISION_FILES = ("vision_weight.q4nx", "vision_weights.q4nx")
# A pull of a local tag must fail rather than overwrite linked files; .invalid never resolves.
_UNREACHABLE_URL = "https://unsloth-local.invalid/q4nx"
_MAX_DEPTH = 4

# (layers, hidden size, experts) -> the FastFlowLM 1.0.12 tag whose kernels and config.json run it.
_FLM_BASES: dict[str, dict[tuple[int, int, int], str]] = {
    "llama": {
        (16, 2048, 0): "llama3.2:1b",
        (28, 3072, 0): "llama3.2:3b",
        (32, 2560, 0): "nanbeige4.1:3b",
        (32, 4096, 0): "llama3.1:8b",
    },
    "qwen2": {(36, 2048, 0): "qwen2.5-it:3b"},
    "qwen2vl": {(36, 2048, 0): "qwen2.5vl-it:3b"},
    "qwen3": {
        (28, 1024, 0): "qwen3:0.6b",
        (28, 2048, 0): "qwen3:1.7b",
        (36, 2560, 0): "qwen3:4b",
        (36, 4096, 0): "qwen3:8b",
    },
    "qwen3vl": {(36, 2560, 0): "qwen3vl-it:4b"},
    "qwen35": {
        (24, 1024, 0): "qwen3.5:0.8b",
        (24, 2048, 0): "qwen3.5:2b",
        (32, 2560, 0): "qwen3.5:4b",
        (32, 4096, 0): "qwen3.5:9b",
    },
    "qwen35moe": {(40, 2048, 256): "qwen3.6-moe:35b-a3b"},
    "gemma3": {(26, 1152, 0): "gemma3:1b", (34, 2560, 0): "gemma3:4b"},
    "gemma4": {
        (35, 1536, 0): "gemma4-it:e2b",
        (42, 2560, 0): "gemma4-it:e4b",
        (48, 3840, 0): "gemma4-it:12b",
    },
    "lfm2": {(16, 2048, 0): "lfm2:1.2b", (30, 2048, 0): "lfm2:2.6b"},
    "phi3": {(32, 3072, 0): "phi4-mini-it:4b"},
    "gpt-oss": {(24, 2880, 32): "gpt-oss:20b"},
}


@dataclass(frozen = True)
class LocalQ4nx:
    folder: Path
    display: str
    size_bytes: int
    files: tuple[str, ...]
    base: Optional[str]
    problem: Optional[str]

    @property
    def family(self) -> str:
        slug = re.sub(r"[^a-z0-9.]+", "-", self.folder.name.lower()).strip("-.")[:40] or "model"
        digest = hashlib.sha1(os.path.normcase(str(self.folder)).encode("utf-8")).hexdigest()[:8]
        return f"{LOCAL_PREFIX}{slug}-{digest}"

    @property
    def checkpoint(self) -> str:
        return f"{self.family}:{LOCAL_SIZE}"

    @property
    def id(self) -> str:
        # Lemonade names an FLM tag family:size as family-size-FLM.
        return f"{self.family}-{LOCAL_SIZE}-FLM"

    @property
    def vision(self) -> bool:
        return any(name in self.files for name in VISION_FILES)

    @property
    def stamp(self) -> str:
        """Changes when the folder is converted again, so a hidden row comes back with new weights."""
        return weights_stamp(self.folder)


def weights_stamp(folder: Path) -> str:
    try:
        info = (folder / "model.q4nx").stat()
    except OSError:
        return ""
    return f"{os.path.normcase(str(folder))}|{info.st_mtime_ns}|{info.st_size}"


def stamp_is_current(stamp: str) -> bool:
    """Whether the weights a stamp names are still the ones in its folder."""
    folder = stamp.split("|", 1)[0]
    return bool(folder) and weights_stamp(Path(folder)) == stamp


def is_local_checkpoint(checkpoint: str) -> bool:
    return checkpoint.startswith(LOCAL_PREFIX)


def is_local_id(model_id: str) -> bool:
    return model_id.startswith(LOCAL_PREFIX)


def _base_for(folder: Path) -> tuple[Optional[str], Optional[str]]:
    try:
        source = json.loads((folder / SOURCE_FILE).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None, (
            "This folder was converted before Unsloth Studio recorded its source model. "
            "Convert the GGUF to Q4NX again to load it on the NPU."
        )
    arch = str(source.get("architecture") or "").lower()
    try:
        layers = int(source["block_count"]) - int(source.get("nextn_predict_layers") or 0)
        hidden = int(source["embedding_length"])
        experts = int(source.get("expert_count") or 0)
    except (KeyError, TypeError, ValueError):
        return None, "The conversion record does not say the model's size. Convert it again."
    base = _FLM_BASES.get(arch, {}).get((layers, hidden, experts))
    if base is None:
        return None, (
            f"FastFlowLM has no AMD NPU kernels for this {arch or 'unknown'} model "
            f"({layers} layers, hidden size {hidden})."
        )
    return base, None


def _describe(folder: Path, root: Path) -> LocalQ4nx:
    present = tuple(
        name
        for name in (*REQUIRED_FILES, *OPTIONAL_FILES, *VISION_FILES)
        if (folder / name).is_file()
    )
    size = sum((folder / name).stat().st_size for name in present)
    missing = [name for name in REQUIRED_FILES if name not in present]
    base, problem = _base_for(folder)
    if missing:
        problem = f"The folder is incomplete: {', '.join(missing)} is missing."
    try:
        display = folder.relative_to(root).as_posix()
    except ValueError:
        display = folder.name
    return LocalQ4nx(
        folder = folder,
        display = display,
        size_bytes = size,
        files = present,
        base = base,
        problem = problem,
    )


def find_local_models(root: Path) -> list[LocalQ4nx]:
    """Folders under root holding a model.q4nx, without following links out of it."""
    from utils.paths.storage_roots import within_account

    found: list[LocalQ4nx] = []
    if not root.is_dir():
        return found
    base_depth = len(root.parts)
    for current, dirs, files in os.walk(root, followlinks = False):
        path = Path(current)
        # Hidden folders are staged_output's in-progress builds.
        dirs[:] = sorted(
            d
            for d in dirs
            if not d.startswith(".") and not os.path.islink(path / d) and not _is_junction(path / d)
        )
        if "model.q4nx" in files:
            dirs[:] = []
            if within_account(path):
                found.append(_describe(path, root))
        elif len(path.parts) - base_depth >= _MAX_DEPTH:
            dirs[:] = []
    return sorted(found, key = lambda model: model.display.lower())


def _is_junction(path: Path) -> bool:
    is_junction = getattr(os.path, "isjunction", None)
    return bool(is_junction and is_junction(path))


def account_local_models() -> list[LocalQ4nx]:
    """The acting account's converted folders; another account's exports root is never read."""
    from utils.paths.storage_roots import exports_root
    return find_local_models(exports_root())


def find_flm_list(flm_binary: Path) -> Optional[Path]:
    """The model_list.json FastFlowLM itself would read when FLM_CONFIG_PATH is unset."""
    for candidate in (
        flm_binary.parent / "model_list.json",
        flm_binary.parent.parent / "share" / "flm" / "model_list.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def base_entry(listing: dict[str, Any], tag: str) -> Optional[dict[str, Any]]:
    family, _, size = tag.partition(":")
    entry = (listing.get("models") or {}).get(family, {}).get(size)
    return entry if isinstance(entry, dict) else None


def staged_folder(flm_model_dir: Path, listing: dict[str, Any], model: LocalQ4nx) -> Path:
    return flm_model_dir / str(listing.get("model_path") or "models") / model.family


def link_files(model: LocalQ4nx, staged: Path) -> None:
    """Hard links where the volume allows (no copy of the weights), else copies."""
    staged.mkdir(parents = True, exist_ok = True)
    keep = set(model.files) | {"config.json"}
    for child in staged.iterdir():
        if child.name not in keep:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors = True)
            else:
                child.unlink(missing_ok = True)
    for name in model.files:
        source, target = model.folder / name, staged / name
        try:
            if target.exists() and os.path.samefile(source, target):
                continue
        except OSError:
            pass
        target.unlink(missing_ok = True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def local_config(base_config: dict[str, Any], model: LocalQ4nx) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    # FastFlowLM turns its vision tower on from this key; a text-only conversion has no weights.
    vision_file = next((name for name in VISION_FILES if name in model.files), None)
    if vision_file is None:
        config.pop("vision_model_weight", None)
    elif config.get("vision_model_weight"):
        config["vision_model_weight"] = vision_file
    return config


def local_entry(
    base: dict[str, Any], model: LocalQ4nx, flm_version: Optional[str]
) -> dict[str, Any]:
    entry = copy.deepcopy(base)
    for key in ("preferred_source", "file_sources", "base_model", "ms_url", "modified_at"):
        entry.pop(key, None)
    url = f"{_UNREACHABLE_URL}/{model.family}"
    labels = [label for label in base.get("label") or [] if model.vision or label != "vision"]
    entry.update(
        {
            "name": model.family,
            # Lemonade reads config.json from FLM_MODEL_PATH/models/<url's last segment>.
            "url": url,
            "file_url": url,
            "files": ["config.json", *model.files],
            "vlm": model.vision,
            "label": labels,
            # "size" stays the base's: engines key behaviour off it (Qwen3.5 enables tools above 0.8B).
            # The engine finds its kernels by this folder name; the base's are the ones that fit.
            "xclbin_folder": base.get("xclbin_folder") or base["name"],
        }
    )
    if flm_version:
        entry["flm_min_version"] = flm_version
    return entry
