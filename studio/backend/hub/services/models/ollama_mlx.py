# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ollama tags in its MLX format (one safetensors blob per tensor) as standard MLX checkpoints.

A load materializes a directory beside the tag's other links: tensor blobs are APFS clones, with
the safetensors header renamed in place where Ollama names the quantization companions, and the
smallest tensors are packed into one real shard, because mlx-lm sets a 2048 open-file soft limit
and MLX keeps every safetensors file open. Each build is keyed by the manifest;
the tag is identified by a stable path through an ``mlx`` link switched to the current build."""

from __future__ import annotations

import collections
import ctypes
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import uuid
from pathlib import Path
from typing import Optional

from loggers import get_logger

from hub.schemas.inventory import LocalModelInfo
from hub.services.models.common import _capabilities_for_format, _local_inventory_id

logger = get_logger(__name__)

TENSOR_MEDIA_TYPE = "application/vnd.ollama.image.tensor"
_JSON_MEDIA_TYPE = "application/vnd.ollama.image.json"
# The renderer, parser and template layers describe Ollama's own prompt handling; Unsloth Studio renders
# with the checkpoint's chat template instead. `image.adapter` changes weights, and `image.system` /
# `image.messages` change every conversation, which this path does not carry over, so neither is here.
_ADMITTED_MEDIA_TYPES = frozenset(
    {
        TENSOR_MEDIA_TYPE,
        _JSON_MEDIA_TYPE,
        "application/vnd.ollama.image.license",
        "application/vnd.ollama.image.template",
        "application/vnd.ollama.image.params",
    }
)
# Drafters ride in the same tensor set (Gemma4's assistant, Qwen3.6's MTP head).
# The assistant becomes a checkpoint of its own; an MTP head stays where mlx-vlm's splitter reads it.
_ASSISTANT_PREFIX = "draft."
_DRAFTER_PREFIXES = (_ASSISTANT_PREFIX, "mtp.")
_ASSISTANT_DIR = "draft"
_GENERATED_JSON = frozenset({"config.json", f"{_ASSISTANT_DIR}/config.json"})
_DROPPED_JSON = frozenset({"hf_quant_config.json"})
# Ollama's names for an MLX-packed weight's companions (create/prequant.go), and MLX's.
_COMPANIONS = {".scale": ".scales", ".bias": ".biases", ".global_scale": ".weight_scale_2"}
# Ollama quant_type -> MLX (group_size, bits, mode), as mlx/quant/quant.go maps them.
_QUANT_TYPES = {
    "nvfp4": (16, 4, "nvfp4"),
    "mxfp4": (32, 4, "mxfp4"),
    "mxfp8": (32, 8, "mxfp8"),
    "int4": (64, 4, "affine"),
    "int8": (64, 8, "affine"),
}
_PACKED_TENSOR_BYTES = 1 << 20
_TAG_FILE = "ollama_tag.txt"
# Part of a build's key, so a change to what a build holds rebuilds it.
_BUILD_LAYOUT = b"2"
_CURRENT_BUILD_LINK = "mlx"
# Well under the 2048 soft limit mlx-lm sets, leaving room for the process's other files.
_MAX_TENSOR_FILES = 1024
_MAX_HEADER_BYTES = 100 << 20
_BUNDLED_TEMPLATES = {"gemma4": ("gemma-4-edge.jinja", "gemma-4.jinja")}


def mlx_host() -> bool:
    from utils.hardware.hardware import is_apple_silicon
    return is_apple_silicon()


def is_mlx_manifest(layers: list) -> bool:
    return any(
        isinstance(layer, dict) and layer.get("mediaType") == TENSOR_MEDIA_TYPE for layer in layers
    )


def _json_layers(layers: list) -> dict[str, dict]:
    return {
        layer["name"]: layer
        for layer in layers
        if isinstance(layer, dict)
        and layer.get("mediaType") == _JSON_MEDIA_TYPE
        and isinstance(layer.get("name"), str)
    }


def layout_problem(config: dict, layers: list) -> Optional[str]:
    """Why Unsloth Studio cannot serve this MLX tag as a chat model, or None."""
    if config.get("model_format") != "safetensors":
        return "it is not a safetensors model"
    if "completion" not in (config.get("capabilities") or ()):
        return "it is not a chat model"
    unsupported = sorted(
        {
            str(layer.get("mediaType")) if isinstance(layer, dict) else "<invalid layer>"
            for layer in layers
            if not isinstance(layer, dict) or layer.get("mediaType") not in _ADMITTED_MEDIA_TYPES
        }
    )
    if unsupported:
        return f"it has layers Unsloth Studio cannot preserve: {', '.join(unsupported)}"
    json_layers = _json_layers(layers)
    if "config.json" not in json_layers:
        return "it has no config.json"
    # A non-JSON name (``model.py``) would land in the checkpoint as code a config could run.
    if any(not name.endswith(".json") for name in json_layers):
        return "it has a JSON layer whose name is not a .json file"
    if not all(
        isinstance(layer.get("name"), str)
        for layer in layers
        if layer.get("mediaType") == TENSOR_MEDIA_TYPE
    ):
        return "a tensor layer has no name"
    return None


def model_info(
    *,
    repo_name: str,
    tag: str,
    checkpoint: str,
    tag_file: Path,
    config: dict,
    layers: list,
    updated_at: Optional[float],
) -> Optional[LocalModelInfo]:
    """The picker row for an MLX tag; None off Apple Silicon or for a tag Unsloth Studio cannot serve."""
    if not mlx_host():
        return None
    problem = layout_problem(config, layers)
    if problem is not None:
        logger.debug("Skipping Ollama MLX tag %s: %s", tag_file, problem)
        return None
    size = sum(
        layer.get("size") or 0
        for layer in layers
        if layer.get("mediaType") == TENSOR_MEDIA_TYPE
        and not layer["name"].startswith(_DRAFTER_PREFIXES)
    )
    file_type = config.get("file_type")
    model_id = f"ollama/{repo_name}:{tag}"
    capabilities = _capabilities_for_format("safetensors", "ollama").model_copy(
        update = {"can_train": False, "supports_lora": False}
    )
    return LocalModelInfo(
        id = checkpoint,
        inventory_id = _local_inventory_id("ollama", "safetensors", model_id),
        load_id = checkpoint,
        model_id = model_id,
        display_name = f"{repo_name}:{tag} (MLX{f' {file_type}' if file_type else ''})",
        path = str(tag_file),
        size_bytes = size if isinstance(size, int) else 0,
        source = "ollama",
        updated_at = updated_at,
        model_format = "safetensors",
        artifact_kind = "transformers_model",
        runtime = "transformers",
        capabilities = capabilities,
    )


def _read_header(path: Path) -> tuple[int, dict]:
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        if size > _MAX_HEADER_BYTES:
            raise ValueError(f"safetensors header of {path.name} is too large")
        header = json.loads(f.read(size))
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header of {path.name} is not an object")
    return size, header


def _mlx_header(header: dict) -> tuple[dict, dict[str, tuple]]:
    """*header* with companions renamed to MLX's and no metadata, and each packed module's
    (quant_type, group_size)."""
    packed = {
        key
        for key, value in header.items()
        if key != "__metadata__"
        and isinstance(value, dict)
        and value.get("dtype") == "U32"
        and f"{key}.scale" in header
    }
    metadata = header.get("__metadata__") if isinstance(header.get("__metadata__"), dict) else {}
    renamed = {}
    for key, value in header.items():
        if key == "__metadata__":
            continue
        for suffix, mlx_suffix in _COMPANIONS.items():
            base = key[: -len(suffix)]
            if key.endswith(suffix) and base in packed:
                key = base.removesuffix(".weight") + mlx_suffix
                break
        renamed[key] = value
    quant = {
        key.removesuffix(".weight"): (
            str(metadata.get(f"{key}.quant_type") or metadata.get("quant_type") or "").lower(),
            metadata.get(f"{key}.group_size") or metadata.get("group_size"),
        )
        for key in packed
    }
    return renamed, quant


def _clonefile(source: Path, target: Path) -> None:
    libc = ctypes.CDLL("libc.dylib", use_errno = True) if sys.platform == "darwin" else None
    if libc is None or libc.clonefile(os.fsencode(str(source)), os.fsencode(str(target)), 0) != 0:
        error = ctypes.get_errno() if libc is not None else 0
        raise ValueError(
            "Loading an Ollama MLX model needs copy-on-write clones of its blobs (APFS), which "
            f"this volume does not support ({os.strerror(error) if error else 'not macOS'})"
        )


def _write_cloned_with_header(source: Path, target: Path, size: int, header: dict) -> None:
    text = json.dumps(header, separators = (",", ":")).encode()
    if len(text) > size:
        raise ValueError(f"renamed safetensors header of {source.name} no longer fits its blob")
    _clonefile(source, target)
    # Same length, padded with spaces as the format allows, so the data offsets do not move.
    with open(target, "r+b") as f:
        f.seek(8)
        f.write(text + b" " * (size - len(text)))


def _write_packed_shard(target: Path, entries: list[tuple[Path, int, dict]]) -> list[str]:
    """One safetensors file holding every tensor of *entries*, copied from their blobs."""
    header, chunks, offset = {}, [], 0
    for blob, size, renamed in entries:
        for name, info in renamed.items():
            start, end = info["data_offsets"]
            chunks.append((blob, 8 + size + start, end - start))
            header[name] = {**info, "data_offsets": [offset, offset + end - start]}
            offset += end - start
    text = json.dumps(header, separators = (",", ":")).encode()
    text += b" " * (-len(text) % 8)
    with open(target, "wb") as out:
        out.write(struct.pack("<Q", len(text)))
        out.write(text)
        for blob, start, length in chunks:
            with open(blob, "rb") as f:
                f.seek(start)
                data = f.read(length)
            if len(data) != length:
                raise ValueError(f"Ollama blob {blob.name} is truncated")
            out.write(data)
    return list(header)


def _quantization(quant: dict[str, tuple]) -> Optional[dict]:
    """The checkpoint's most common packing as its default; the loader reads each layer's own."""
    specs = []
    for quant_type, group_size in quant.values():
        if quant_type not in _QUANT_TYPES:
            raise ValueError(
                f"Ollama MLX quantization {quant_type or '<missing>'!r} is not supported"
            )
        default_group, bits, mode = _QUANT_TYPES[quant_type]
        specs.append((int(group_size or default_group), bits, mode))
    if not specs:
        return None
    group_size, bits, mode = collections.Counter(specs).most_common(1)[0][0]
    return {"group_size": group_size, "bits": bits, "mode": mode}


def _bundled_chat_template(ollama_config: dict, config: dict) -> Optional[str]:
    templates = _BUNDLED_TEMPLATES.get(ollama_config.get("renderer"))
    if templates is None:
        return None
    text_config = (
        config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    )
    # Gemma 4's per-layer-embedding (E2B / E4B) models omit the empty thought block when thinking is off.
    edge = bool(text_config.get("hidden_size_per_layer_input"))
    from core.inference.chat_templates import load_bundled_chat_template

    return load_bundled_chat_template(templates[0] if edge else templates[1])


def _contained(root: Path, name: str) -> Optional[Path]:
    """*name* (a manifest field) as a path under *root*, or None when it would escape it."""
    if (
        not name
        or "\\" in name
        or name.startswith("/")
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        return None
    return root / name


def _write_new(path: Path, text: str) -> None:
    # Exclusive create: a manifest layer linked under this name (in any case) must not redirect it into a blob.
    with open(path, "x", encoding = "utf-8") as f:
        f.write(text)


def _write_tensors(
    directory: Path,
    blobs_dir: Path,
    blob_path,
    layers: list,
    strip: str = "",
) -> dict:
    """*layers*' tensors as MLX shards and an index in *directory*, names without *strip*; returns
    each packed module's (quant_type, group_size)."""
    tensors = []
    for layer in layers:
        blob = blob_path(blobs_dir, layer.get("digest"))
        if blob is None or not blob.is_file():
            raise ValueError(f"Ollama blob for {layer['name']} is missing")
        size, header = _read_header(blob)
        renamed, quant = _mlx_header(header)
        renamed = {key.removeprefix(strip): value for key, value in renamed.items()}
        quant = {key.removeprefix(strip): value for key, value in quant.items()}
        tensors.append((blob.stat().st_size, blob, size, header, renamed, quant))
    tensors.sort(key = lambda item: item[0])
    small = sum(1 for item in tensors if item[0] < _PACKED_TENSOR_BYTES)
    packed_count = max(small, len(tensors) - _MAX_TENSOR_FILES)

    directory.mkdir(parents = True, exist_ok = True)
    weight_map, quant = {}, {}
    shard_names = [f"model-{i + 1:05d}.safetensors" for i in range(len(tensors) - packed_count + 1)]
    if packed_count:
        packed = [(blob, size, renamed) for _, blob, size, _, renamed, _ in tensors[:packed_count]]
        for name in _write_packed_shard(directory / shard_names[0], packed):
            weight_map[name] = shard_names[0]
    for shard, (_, blob, size, header, renamed, _) in zip(shard_names[1:], tensors[packed_count:]):
        # A clone even when unchanged: a retained build must survive Ollama reclaiming the blob.
        if list(renamed) == [key for key in header if key != "__metadata__"]:
            _clonefile(blob, directory / shard)
        else:
            _write_cloned_with_header(blob, directory / shard, size, renamed)
        weight_map.update(dict.fromkeys(renamed, shard))
    for *_, item_quant in tensors:
        quant.update(item_quant)
    _write_new(
        directory / "model.safetensors.index.json",
        json.dumps({"metadata": {}, "weight_map": weight_map}),
    )
    return quant


def _write_config(directory: Path, blobs_dir: Path, blob_path, layer: dict, quant: dict) -> dict:
    blob = blob_path(blobs_dir, layer.get("digest"))
    if blob is None:
        raise ValueError(f"Ollama MLX layer {layer['name']!r} is unusable")
    config = json.loads(blob.read_text(encoding = "utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Ollama MLX {layer['name']} is not an object")
    # mlx-vlm imports ``model_file`` even without trust_remote_code.
    if "model_file" in config:
        raise ValueError(f"Ollama MLX {layer['name']} names a model_file to run")
    # Its ModelOpt block describes the source export, not the MLX packing Ollama converted it to.
    config.pop("quantization_config", None)
    quantization = _quantization(quant)
    if quantization is not None:
        config["quantization"] = quantization
    _write_new(directory / "config.json", json.dumps(config, indent = 2))
    return config


def _build(
    directory: Path, blobs_dir: Path, blob_path, manifest: dict, ollama_config: dict
) -> None:
    layers = manifest["layers"]
    tensors = [layer for layer in layers if layer.get("mediaType") == TENSOR_MEDIA_TYPE]
    json_layers = _json_layers(layers)
    assistant = [layer for layer in tensors if layer["name"].startswith(_ASSISTANT_PREFIX)]
    quant = _write_tensors(
        directory,
        blobs_dir,
        blob_path,
        [layer for layer in tensors if not layer["name"].startswith(_ASSISTANT_PREFIX)],
    )
    if f"{_ASSISTANT_DIR}/config.json" not in json_layers:
        assistant = []
    if assistant:
        draft_quant = _write_tensors(
            directory / _ASSISTANT_DIR, blobs_dir, blob_path, assistant, _ASSISTANT_PREFIX
        )
        _write_config(
            directory / _ASSISTANT_DIR,
            blobs_dir,
            blob_path,
            json_layers[f"{_ASSISTANT_DIR}/config.json"],
            draft_quant,
        )

    for name, layer in json_layers.items():
        if name in _GENERATED_JSON or name in _DROPPED_JSON:
            continue
        if name.startswith(f"{_ASSISTANT_DIR}/") and not assistant:
            continue
        target, blob = _contained(directory, name), blob_path(blobs_dir, layer.get("digest"))
        if target is None or blob is None or not blob.is_file():
            raise ValueError(f"Ollama MLX layer {name!r} is unusable")
        target.parent.mkdir(parents = True, exist_ok = True)
        target.symlink_to(blob)

    config = _write_config(directory, blobs_dir, blob_path, json_layers["config.json"], quant)
    tokenizer_config = directory / "tokenizer_config.json"
    has_template = (directory / "chat_template.jinja").exists() or (
        tokenizer_config.exists()
        and bool(json.loads(tokenizer_config.read_text(encoding = "utf-8")).get("chat_template"))
    )
    if not has_template:
        template = _bundled_chat_template(ollama_config, config)
        if template is not None:
            _write_new(directory / "chat_template.jinja", template)


def _checkpoint_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id.removeprefix("ollama/")).strip(".-") or "model"


def _checkpoint_model_id(path: Path) -> Optional[str]:
    """The ``ollama/<repo>:<tag>`` id a finished build at *path* was made for, else None."""
    try:
        return (path / _TAG_FILE).read_text(encoding = "utf-8").strip() or None
    except (OSError, UnicodeDecodeError):
        return None


def checkpoint_path(tag_dir: Path, model_id: str) -> Path:
    """The tag's checkpoint as clients and the loaded model name it, the same across re-pulls; its
    last component is the tag, which clients read the name and name-keyed defaults from."""
    return tag_dir / _CURRENT_BUILD_LINK / _checkpoint_name(model_id)


def materialize(
    tag_dir: Path,
    blobs_dir: Path,
    blob_path,
    manifest_bytes: bytes,
    ollama_config: dict,
    model_id: str,
) -> str:
    """Builds the checkpoint for this manifest once and points ``checkpoint_path`` at it. The
    build that was current before it stays, since a resident model reads its shards by path; older
    ones go, as clones keep a re-pulled or removed blob's bytes alive. The caller holds the tag's
    materialization lock."""
    manifest = json.loads(manifest_bytes)
    problem = layout_problem(ollama_config, manifest.get("layers") or [])
    if problem is not None:
        raise ValueError(f"Unsloth Studio cannot load this Ollama MLX model: {problem}")
    name = _checkpoint_name(model_id)
    build = tag_dir / f"mlx-{hashlib.sha256(_BUILD_LAYOUT + manifest_bytes).hexdigest()[:16]}"
    if _checkpoint_model_id(build / name) != model_id:
        tag_dir.mkdir(parents = True, exist_ok = True)
        staging = tag_dir / f".{build.name}.tmp-{uuid.uuid4().hex[:8]}"
        (staging / name).mkdir(parents = True)
        try:
            _build(staging / name, blobs_dir, blob_path, manifest, ollama_config)
            _write_new(staging / name / _TAG_FILE, model_id)
            if build.exists():
                shutil.rmtree(build)
            os.replace(staging, build)
        except BaseException:
            shutil.rmtree(staging, ignore_errors = True)
            raise
    link = tag_dir / _CURRENT_BUILD_LINK
    if not link.is_symlink() or os.readlink(link) != build.name:
        # A build's mtime records when it last became current; the prune keeps the latest other one.
        os.utime(build)
        staged_link = tag_dir / f".{_CURRENT_BUILD_LINK}-link-{uuid.uuid4().hex[:8]}"
        staged_link.symlink_to(build.name)
        os.replace(staged_link, link)
    earlier = [path for path in tag_dir.iterdir() if path.name.startswith("mlx-") and path != build]
    previous = max(earlier, key = lambda path: path.stat().st_mtime_ns, default = None)
    for stale in tag_dir.iterdir():
        if stale.name.startswith(("mlx-", ".mlx-")) and stale not in (build, previous):
            shutil.rmtree(stale, ignore_errors = True)
    return str(checkpoint_path(tag_dir, model_id))
