# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which GGUF architectures and shapes the installed FastFlowLM runs and the installed Q4NX
converter converts. Read from their files; neither is ever started."""

from __future__ import annotations

import ast
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from loggers import get_logger

logger = get_logger(__name__)

# Hugging Face model_type -> GGUF general.architecture, for the catalog models' config.json.
_GGUF_ARCH = {
    "llama": "llama",
    "qwen2": "qwen2",
    "qwen2_5_vl": "qwen2vl",
    "qwen3": "qwen3",
    "qwen3_vl": "qwen3vl",
    "qwen3_5": "qwen35",
    "qwen3_5_text": "qwen35",
    "qwen3_5_moe": "qwen35moe",
    "gemma3": "gemma3",
    "gemma3_text": "gemma3",
    "gemma3_text_only": "gemma3",
    "gemma4": "gemma4",
    "gemma4_text": "gemma4",
    "gemma4_unified": "gemma4",
    "lfm2": "lfm2",
    "lfm3": "lfm3",
    "phi3": "phi3",
    "gpt_oss": "gpt-oss",
}
# FastFlowLM's Qwen3-VL config.json is its text tower's, model_type "qwen3".
_VISION_ARCH = {"qwen3": "qwen3vl"}
_SKIPPED_LABELS = frozenset({"embeddings", "transcription"})
_PLATFORM = "aie2p"
_FETCH_TIMEOUT_S = 15.0

Fetch = Callable[[str], Optional[dict]]


def _fetch_json(url: str) -> Optional[dict]:
    try:
        response = httpx.get(url, follow_redirects = True, timeout = _FETCH_TIMEOUT_S)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("Could not read %s: %s", url, exc)
        return None
    return data if isinstance(data, dict) else None


def _config_url(entry: dict) -> str:
    base = str(entry["url"]).rstrip("/")
    # FastFlowLM's own rule: a url naming a revision already ends in /resolve/<revision>.
    return f"{base if '/resolve' in base else base + '/resolve/main'}/config.json"


def _geometry(config: dict) -> Optional[dict]:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}

    def get(key):
        return config.get(key, text.get(key))

    layers, hidden = get("num_hidden_layers"), get("hidden_size")
    if not (isinstance(layers, int) and isinstance(hidden, int)):
        return None
    experts = get("num_experts") or get("num_local_experts") or 0
    return {
        "model_type": str(config.get("model_type") or ""),
        "layers": layers,
        "hidden": hidden,
        "experts": experts if isinstance(experts, int) else 0,
    }


def _entries(listing: dict):
    for family, sizes in (listing.get("models") or {}).items():
        for size, entry in (sizes or {}).items():
            if not isinstance(entry, dict) or _PLATFORM not in (
                entry.get("supported_platforms") or ()
            ):
                continue
            # A DFlash checkpoint's config.json describes its drafter, not the model.
            if "dflash" in family or _SKIPPED_LABELS.intersection(entry.get("label") or ()):
                continue
            yield f"{family}:{size}", size, entry


def flm_shapes(
    flm_binary: Optional[Path],
    flm_model_path: Path,
    cache_path: Path,
    fetch: Fetch = _fetch_json,
) -> Optional[list[dict]]:
    """FastFlowLM's runnable shapes, one row per (architecture, layers, hidden, experts); None
    when FastFlowLM is not installed or a catalog model's config.json could not be read (offline),
    since a partial table would refuse sizes it runs. A model's geometry comes from its downloaded
    config.json, else its Hugging Face repo, cached on disk per model_list.json."""
    from core.inference.npu_local_models import find_flm_list

    listing_path = find_flm_list(flm_binary) if flm_binary else None
    if listing_path is None:
        return None
    try:
        raw = listing_path.read_bytes()
        listing = json.loads(raw)
    except (OSError, ValueError):
        return None
    generation = hashlib.sha256(raw).hexdigest()
    try:
        cached = json.loads(cache_path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        cached = {}
    known = cached.get("configs", {}) if cached.get("generation") == generation else {}

    entries = list(_entries(listing))

    def geometry(item) -> tuple[str, Optional[dict]]:
        checkpoint, _, entry = item
        if checkpoint in known:
            return checkpoint, known[checkpoint]
        local = (
            flm_model_path
            / str(listing.get("model_path") or "")
            / str(entry.get("name"))
            / "config.json"
        )
        try:
            config = json.loads(local.read_text(encoding = "utf-8"))
        except (OSError, ValueError):
            config = fetch(_config_url(entry)) if entry.get("url") else None
        return checkpoint, _geometry(config) if isinstance(config, dict) else None

    with ThreadPoolExecutor(max_workers = 8) as pool:
        found = dict(pool.map(geometry, entries))
    fresh = {k: v for k, v in found.items() if v is not None}
    if fresh != known:
        try:
            cache_path.parent.mkdir(parents = True, exist_ok = True)
            cache_path.write_text(
                json.dumps({"generation": generation, "configs": fresh}), encoding = "utf-8"
            )
        except OSError as exc:
            logger.debug("Could not cache FastFlowLM model shapes: %s", exc)
    if len(fresh) < len(found):
        return None

    shapes: dict[tuple, dict] = {}
    word_sized: set[str] = set()
    for checkpoint, size, entry in entries:
        geo = found.get(checkpoint)
        arch = _GGUF_ARCH.get(geo["model_type"]) if geo else None
        if arch is None:
            continue
        if entry.get("vlm"):
            arch = _VISION_ARCH.get(arch, arch)
        key = (arch, geo["layers"], geo["hidden"], geo["experts"])
        row = shapes.setdefault(
            key,
            {
                "architecture": arch,
                "layers": geo["layers"],
                "hidden": geo["hidden"],
                "experts": geo["experts"],
                "checkpoints": [],
                "names": [],
                "sizes": [],
            },
        )
        name = str(entry.get("name") or checkpoint).removesuffix("-NPU2")
        # Sizes are matched against repo names, so a word size ("turbo") is left out.
        size = size.lower() if any(ch.isdigit() for ch in size) else ""
        for field, value in (("checkpoints", checkpoint), ("names", name), ("sizes", size)):
            if value and value not in row[field]:
                row[field].append(value)
        if not size:
            word_sized.add(name)
    # The plainest name leads ("Qwen3-8B", not "DeepSeek-R1-0528-Qwen3-8B"; not "D3-Turbo").
    for row in shapes.values():
        row["names"].sort(key = lambda n: (n in word_sized, len(n)))
    return [shapes[key] for key in sorted(shapes, key = lambda k: (k[0], k[2], k[1], k[3]))]


def _literal_dict(tree: ast.Module, name: str) -> Optional[ast.Dict]:
    for node in tree.body:
        target = (
            node.targets[0]
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            else (node.target if isinstance(node, ast.AnnAssign) else None)
        )
        if isinstance(target, ast.Name) and target.id == name and isinstance(node.value, ast.Dict):
            return node.value
    return None


def _enum_members(tree: ast.Module, name: str) -> set[str]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return {
                t.id
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                for t in stmt.targets
                if isinstance(t, ast.Name)
            }
    return set()


def _arch_names(package: Path) -> tuple[dict[str, list[str]], set[str]]:
    """ModelArchNames as {member: [aliases]} and the ModelArch members, parsed, never imported."""
    tree = ast.parse((package / "constants.py").read_text(encoding = "utf-8"))
    table = _literal_dict(tree, "ModelArchNames")
    if table is None:
        raise ValueError("constants.py has no ModelArchNames table")
    names = {
        key.attr: [str(v) for v in ast.literal_eval(value)]
        for key, value in zip(table.keys, table.values)
        if isinstance(key, ast.Attribute)
    }
    return names, _enum_members(tree, "ModelArch")


def _qwen35_geometry(package: Path, members: set[str]) -> list[list[int]]:
    """The (layers, hidden) sizes _resolve_qwen35 accepts, with its QWEN35_27B add-on when present."""
    tree = ast.parse((package / "model_converter.py").read_text(encoding = "utf-8"))
    table = _literal_dict(tree, "_QWEN35_GEOMETRY")
    sizes = [list(ast.literal_eval(key)) for key in table.keys] if table is not None else []
    if "QWEN35_27B" in members and [64, 5120] not in sizes:
        sizes.append([64, 5120])
    return sizes


def converter_support(
    root: Path,
    installer,
    install: bool = True,
) -> Optional[dict]:
    """What the pinned converters accept, read from their installed source; installs them first
    (small, hash-checked) when missing. None when they cannot be read."""
    try:
        packages, commits = {}, {}
        for name in ("q4nx", "q4k"):
            pins = installer.load_pins(name = name)
            script = installer.installed_converter(root, pins)
            if script is None and install:
                script = installer.install(root, name = name)
            if script is None:
                return None
            packages[name] = script.parent / "q4nx"
            commits[name] = pins["commit"]
        q4nx_names, _ = _arch_names(packages["q4nx"])
        q4k_names, q4k_members = _arch_names(packages["q4k"])
        qwen35 = _qwen35_geometry(packages["q4k"], q4k_members)
    except Exception as exc:  # noqa: BLE001 -- an unreadable converter reads as unknown, never an error
        logger.warning("Could not read the Q4NX converter's architectures: %s", exc)
        return None

    # converter_for_architecture matches general.architecture exactly, then general.basename by prefix.
    def routed(names: dict[str, list[str]], converter: str) -> set[str]:
        aliases = {a.lower() for row in names.values() for a in row}
        return {a for a in aliases if installer.converter_for_architecture(a) == converter}

    q4nx_aliases = routed(q4nx_names, "q4nx")
    architectures = q4nx_aliases | routed(q4k_names, "q4k")
    # A prefix that is an architecture misroutes (Qwen2-VL / Qwen3-MoE to dense), so only name-only aliases count.
    basenames = q4nx_aliases - set(_GGUF_ARCH.values())
    if qwen35:
        architectures.add("qwen35")
    return {
        "architectures": sorted(architectures),
        "basenames": sorted(basenames),
        "qwen35_shapes": sorted(qwen35),
        "commits": commits,
    }


def build_compat(shapes: Optional[list[dict]], converter: Optional[dict]) -> dict:
    convertible = set(converter["architectures"]) if converter else set()
    basenames = tuple(converter["basenames"]) if converter else ()
    qwen35 = {tuple(s) for s in converter["qwen35_shapes"]} if converter else set()

    def converts(arch: str, shape: Optional[dict] = None) -> bool:
        if arch == "qwen35" and shape is not None:
            return (shape["layers"], shape["hidden"]) in qwen35
        # qwen2vl, for one, is matched by the model's name ("Qwen2.5-VL"), not its architecture.
        names = shape["names"] if shape else ()
        return arch in convertible or any(n.lower().startswith(basenames) for n in names)

    # A Nanbeige GGUF says "nanbeige"; FastFlowLM's config.json for it says llama.
    aliased = [
        {**s, "architecture": "nanbeige"}
        for s in shapes or []
        if s["architecture"] == "llama" and any("nanbeige" in n.lower() for n in s["names"])
    ]
    rows = [
        {**shape, "runs": True, "converts": converts(shape["architecture"], shape)}
        for shape in [*(shapes or []), *aliased]
    ]
    runnable = {row["architecture"] for row in rows}
    # Converter aliases that are basename prefixes ("medgemma", "qwen3.5-9b") are not rows.
    known = runnable | (convertible & set(_GGUF_ARCH.values()))
    return {
        "fastflowlm_known": shapes is not None,
        "converter_known": converter is not None,
        "converter": converter,
        "shapes": rows,
        "architectures": [
            {"architecture": arch, "runs": arch in runnable, "converts": converts(arch)}
            for arch in sorted(known)
        ],
    }


_lock = threading.Lock()
_memo: dict[tuple, dict] = {}


def _stamp(path: Optional[Path]) -> Optional[tuple[int, int]]:
    try:
        info = path.stat() if path else None
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size) if info else None


def current(install_converter: bool = True) -> dict:
    """The compat table for this install, computed once per installed FastFlowLM and converter."""
    from core.export.q4nx import _installer
    from core.inference.npu_backend import get_npu_backend, npu_model_dir
    from utils.paths.storage_roots import studio_root

    backend = get_npu_backend()
    # No supported NPU: nothing is installed or fetched, every model reads as unknown.
    if not backend.hardware().get("supported"):
        return build_compat(None, None)
    flm_binary = backend._flm_binary()
    root = studio_root() / "q4nx_converter"
    installer = _installer()
    from core.inference.npu_local_models import find_flm_list

    key = (
        str(flm_binary),
        _stamp(find_flm_list(flm_binary) if flm_binary else None),
        tuple(installer.load_pins(name = n)["commit"] for n in ("q4nx", "q4k")),
    )
    with _lock:
        hit = _memo.get(key)
        if hit is not None:
            return hit
        shapes = flm_shapes(flm_binary, npu_model_dir(), backend.root / "npu_compat_shapes.json")
        result = build_compat(shapes, converter_support(root, installer, install = install_converter))
        # An unknown half is retried next time (offline, or not enabled yet).
        if result["fastflowlm_known"] and result["converter_known"]:
            _memo.clear()
            _memo[key] = result
        return result


def runnable_shapes() -> Optional[dict[str, dict[tuple[int, int, int], str]]]:
    """{architecture: {(layers, hidden, experts): model name}} FastFlowLM runs; None when unknown."""
    compat = current(install_converter = False)
    if not compat["fastflowlm_known"]:
        return None
    table: dict[str, dict[tuple[int, int, int], str]] = {}
    for row in compat["shapes"]:
        shape = (row["layers"], row["hidden"], row["experts"])
        table.setdefault(row["architecture"], {})[shape] = row["names"][0]
    return table
