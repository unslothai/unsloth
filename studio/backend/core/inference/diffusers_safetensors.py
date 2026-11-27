# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""When a diffusers load may pass ``use_safetensors``, and which repo files a pipeline download skips.

``True`` turns the silent pickle fallback into an error, so it is passed only when every weight-bearing component of
the folder ships ``.safetensors``; an uninspectable folder keeps the library default. ``False`` is passed only to
reuse cached ``.bin`` weights whose sha256 matches the Hub (or offline, where the default would refuse the load).
No torch, never raises."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

_PICKLE_WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".bin", ".pkl", ".pickle")


def is_pickle_weight(name: Optional[Any]) -> bool:
    return str(name or "").strip().lower().endswith(_PICKLE_WEIGHT_SUFFIXES)


_MANIFESTS = ("model_index.json", "modular_model_index.json")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle")
# Pickle containers no diffusers / transformers component load reads: always skipped in a pipeline download.
_NEVER_READ_PICKLES = (".pt", ".pth", ".ckpt", ".pkl", ".pickle")


# The weight names a component load resolves (diffusers ``diffusion_pytorch_model``, transformers ``model``), single or
# sharded. dtype-variant twins (``model.fp16.safetensors``) never match, and any other ``.safetensors``
# (``adapter_model.safetensors``, an aux file) is not the component's weights.
_MODEL_SAFETENSORS = re.compile(
    r"^(diffusion_pytorch_model|model)(-\d+-of-\d+)?\.safetensors(\.index\.json)?$"
)


def _is_model_safetensors(name: str) -> bool:
    return bool(_MODEL_SAFETENSORS.match(name.rsplit("/", 1)[-1]))


# The pickle twins of those names (``diffusion_pytorch_model.bin``, ``pytorch_model-00001-of-00002.bin``).
_MODEL_BIN = re.compile(r"^(diffusion_pytorch_model|pytorch_model)(-\d+-of-\d+)?\.bin$")


def _is_model_bin(name: str) -> bool:
    return bool(_MODEL_BIN.match(name.rsplit("/", 1)[-1]))


def _snapshot_identity(root: Path) -> Optional[tuple]:
    """``(repo_id, commit)`` of a Hub cache snapshot folder (``.../models--a--b/snapshots/<commit>``), else None."""
    try:
        parts = Path(os.path.realpath(str(root))).parts
        if len(parts) < 3 or parts[-2] != "snapshots" or not parts[-3].startswith("models--"):
            return None
        return parts[-3][len("models--") :].replace("--", "/"), parts[-1]
    except Exception:  # noqa: BLE001
        return None


def _hub_offline() -> bool:
    try:
        from huggingface_hub import constants
        return bool(constants.HF_HUB_OFFLINE)
    except Exception:  # noqa: BLE001
        return True


def verified_bin_components(
    root: Any,
    components: Iterable[str],
    *,
    token: Optional[str] = None,
    offline: bool = False,
    revision: Optional[str] = None,
) -> set:
    """The ``components`` of the cached snapshot ``root`` whose weights are on disk only as ``.bin`` (no
        ``.safetensors`` weight cached) and every such ``.bin`` hashes to the sha256 the Hub records for it at the
        revision the load will read (``revision``, or the Hub's current ``main`` when the load is unpinned, never the
    cached snapshot's possibly stale commit). Empty for a folder that is not a Hub cache snapshot or on any doubt. ``offline`` (nothing can
        be downloaded, so the alternative to the cached ``.bin`` is a failed load) accepts the cached ``.bin`` unhashed."""
    try:
        from .cached_weight_sha import cached_copy_matches_hub, hub_sha256s, reuse_enabled

        if not reuse_enabled():
            return set()
        ident = _snapshot_identity(Path(str(root)))
        if ident is None:
            return set()
        repo_id, _commit = ident
        offline = offline or _hub_offline()
        wanted: dict = {}
        for component in components:
            folder = Path(str(root)) / component
            if not folder.is_dir() or component_weight_state(folder) is not False:
                continue
            bins = [p.name for p in folder.iterdir() if p.is_file() and _is_model_bin(p.name)]
            if bins:
                wanted[component] = [f"{component}/{b}" for b in bins]
        if not wanted:
            return set()
        if offline:
            return set(wanted)
        shas = hub_sha256s(
            repo_id, [n for names in wanted.values() for n in names], revision = revision, token = token
        )
        return {
            component
            for component, names in wanted.items()
            if all(
                n in shas
                and cached_copy_matches_hub(
                    repo_id, n, str(Path(str(root)) / n), revision = revision, expected = shas[n]
                )
                for n in names
            )
        }
    except Exception:  # noqa: BLE001 - unverified: safetensors stays first
        return set()


def component_weight_state(folder: Any) -> Optional[bool]:
    """True: ``folder`` holds the component's own non-variant ``.safetensors`` weight (or its index); False: it holds
    weights and none of them is that; None: no weight file at all (or unreadable)."""
    try:
        names = [p.name for p in Path(str(folder)).iterdir() if p.is_file()]
    except OSError:
        return None
    weights = [
        n for n in names if n.lower().endswith(_WEIGHT_SUFFIXES) or n.endswith(".index.json")
    ]
    if not weights:
        return None
    # A lone shard without its index is not loadable (an interrupted download): only a whole file or an index counts.
    if any(
        _is_model_safetensors(n) and not re.search(r"-\d+-of-\d+\.safetensors$", n) for n in weights
    ):
        return True
    return False if any(not n.endswith(".index.json") for n in weights) else None


def _manifest_components(root: Path) -> Optional[list]:
    for manifest in _MANIFESTS:
        path = root / manifest
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding = "utf-8"))
        except Exception:  # noqa: BLE001 - unreadable manifest: unknown
            return None
        if not isinstance(data, dict):
            return None
        out = []
        for key, value in data.items():
            if key.startswith("_"):
                continue
            # classic manifests: [library, class]; modular ones: [library, class, {spec}] or nested specs
            if isinstance(value, (list, tuple)) and value and value[0] is not None:
                out.append((key, value[1] if len(value) > 1 else None))
        return out
    return None


# Manifest classes that never hold weights (a scheduler / tokenizer folder is config only).
_WEIGHTLESS_CLASS_SUFFIXES = (
    "Scheduler",
    "Tokenizer",
    "TokenizerFast",
    "Processor",
    "FeatureExtractor",
)


def _weightless(class_name: Any) -> bool:
    return isinstance(class_name, str) and class_name.endswith(_WEIGHTLESS_CLASS_SUFFIXES)


def local_pipeline_safetensors_complete(root: Any, passed: Iterable[str] = ()) -> bool:
    """True when ``root`` is a local diffusers pipeline folder whose every manifest component that holds weights
    (and is not supplied pre-built in ``passed``) has a ``.safetensors`` weight. False when unknown (a weight-bearing
    component missing or without weights, e.g. a partly cached snapshot) or any component is pickle-only."""
    try:
        base = Path(str(root)).expanduser()
        if not base.is_dir():
            return False
    except (OSError, ValueError):
        return False
    components = _manifest_components(base)
    if components is None:
        return False
    skip = set(passed or ())
    seen = False
    for name, class_name in components:
        if name in skip:
            continue
        folder = base / name
        state = component_weight_state(folder) if folder.is_dir() else None
        if state is False:
            return False
        if state is None:
            if _weightless(class_name):
                continue
            return False
        seen = True
    return seen


def _cached_snapshot_dir(
    repo_id: str,
    probe: str,
    cache_dir: Optional[str],
    revision: Optional[str] = None,
) -> Optional[Path]:
    """The cached snapshot folder of ``revision`` (default: the cached ``main``) holding ``repo_id/probe``, or None.
    Offline: reads the cache only."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001
        return None
    for root in dict.fromkeys((cache_dir, None)):
        try:
            hit = try_to_load_from_cache(
                repo_id, probe, cache_dir = root, **({"revision": revision} if revision else {})
            )
        except Exception:  # noqa: BLE001
            continue
        if isinstance(hit, str) and os.path.isfile(hit):
            return Path(hit).parents[len(probe.split("/")) - 1]
    return None


def _takes_use_safetensors(cls: Any, base_name: str) -> bool:
    """Only diffusers' own ``DiffusionPipeline`` / ``ModelMixin`` subclasses are known to accept the kwarg."""
    if cls is None:
        return True
    try:
        import diffusers
        base = getattr(diffusers, base_name, None)
        return isinstance(cls, type) and base is not None and issubclass(cls, base)
    except Exception:  # noqa: BLE001
        return False


def pipeline_safetensors_kwargs(
    source: Any,
    *,
    passed: Iterable[str] = (),
    cache_dir: Optional[str] = None,
    pipeline_cls: Any = None,
) -> dict:
    """``{"use_safetensors": True}`` for a pipeline ``from_pretrained(source)`` whose folder (``source`` itself, or the
    cached snapshot of a Hub id) ships safetensors for every weight-bearing component; ``{}`` otherwise (and for a
    ``pipeline_cls`` that is not a diffusers ``DiffusionPipeline``)."""
    try:
        if not _takes_use_safetensors(pipeline_cls, "DiffusionPipeline"):
            return {}
        text = str(source or "").strip()
        if not text:
            return {}
        if os.path.isdir(os.path.expanduser(text)):
            root: Optional[Path] = Path(text).expanduser()
            # A snapshot folder is read as is: its own commit is the revision the load reads.
            revision = (_snapshot_identity(root) or (None, None))[1]
        else:
            root = None
            # The snapshot the load reads: a pinned ``revision`` among the load's kwargs, else the cached main (whose
            # .bin files are then checked against the Hub's current main, what an unpinned load resolves to).
            revision = passed.get("revision") if isinstance(passed, dict) else None
            for manifest in _MANIFESTS:
                root = _cached_snapshot_dir(text, manifest, cache_dir, revision)
                if root is not None:
                    break
        if root is not None and local_pipeline_safetensors_complete(root, passed):
            return {"use_safetensors": True}
        variant = passed.get("variant") if isinstance(passed, dict) else None
        # A variant load reads ``*.<variant>.*`` files, which the default-name .bin check says nothing about.
        if root is not None and not variant and _snapshot_identity(root) is not None:
            # Every weight component already cached as a Hub-verified .bin: reuse it, download nothing new.
            components = _manifest_components(root) or []
            skip = set(passed or ())
            needed = [
                name
                for name, class_name in components
                if name not in skip and not _weightless(class_name)
            ]
            token = passed.get("token") if isinstance(passed, dict) else None
            offline = bool(passed.get("local_files_only")) if isinstance(passed, dict) else False
            if needed and verified_bin_components(
                root, needed, token = token, offline = offline, revision = revision
            ) == set(needed):
                return {"use_safetensors": False}
    except Exception:  # noqa: BLE001 - unknown keeps the library default
        pass
    return {}


def component_safetensors_kwargs(
    source: Any,
    subfolder: str,
    *,
    cache_dir: Optional[str] = None,
    model_cls: Any = None,
    local_files_only: bool = False,
    revision: Optional[str] = None,
) -> dict:
    """``{"use_safetensors": True}`` for ``from_pretrained(source, subfolder = subfolder)`` when that component folder
    (local, or in the cached snapshot of a Hub id) holds a safetensors weight; ``{}`` otherwise (and for a
    ``model_cls`` that is not a diffusers ``ModelMixin``)."""
    try:
        if not _takes_use_safetensors(model_cls, "ModelMixin"):
            return {}
        text = str(source or "").strip()
        if not text or not subfolder:
            return {}
        if os.path.isdir(os.path.expanduser(text)):
            folder: Optional[Path] = Path(text).expanduser() / subfolder
            revision = (_snapshot_identity(folder.parent) or (None, revision))[1]
        else:
            snap = _cached_snapshot_dir(text, f"{subfolder}/config.json", cache_dir, revision)
            folder = snap / subfolder if snap is not None else None
        if folder is not None and component_weight_state(folder) is True:
            return {"use_safetensors": True}
        if folder is not None and verified_bin_components(
            folder.parent, [subfolder], offline = local_files_only, revision = revision
        ):
            return {"use_safetensors": False}
    except Exception:  # noqa: BLE001
        pass
    return {}


def pipeline_pickle_files(filenames: Iterable[str]) -> list:
    """Pipeline repo files to skip: pickles in a component folder or beside a same-stem root ``.safetensors``, and
    ``.bin`` beside the component's own safetensors. A pickle with no twin is the only copy and stays."""
    names = [n for n in filenames if isinstance(n, str)]
    if not any(n in _MANIFESTS for n in names):
        return []
    st_dirs = {n.rsplit("/", 1)[0] if "/" in n else "" for n in names if _is_model_safetensors(n)}
    st_stems = {
        n[: -len(".safetensors")].lower() for n in names if n.lower().endswith(".safetensors")
    }
    out = []
    for n in names:
        low = n.lower()
        folder = n.rsplit("/", 1)[0] if "/" in n else ""
        if low.endswith(_NEVER_READ_PICKLES):
            if folder or low.rsplit(".", 1)[0] in st_stems:
                out.append(n)
        elif (low.endswith(".bin") or low.endswith(".bin.index.json")) and folder in st_dirs:
            out.append(n)
    return out


def is_never_read_pickle(name: Optional[str]) -> bool:
    """A pickle container (``.pt`` / ``.pth`` / ``.ckpt`` / ``.pkl``) no diffusers component load reads."""
    return bool(name) and is_pickle_weight(name) and str(name).lower().endswith(_NEVER_READ_PICKLES)
