# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared LoRA support for the Studio diffusion backends.

The native sd-cli engine selects adapters by `<lora:NAME:WEIGHT>` prompt tags resolved against a
`--lora-model-dir`; diffusers loads them with `load_lora_weights()` + `set_adapters()`. This
module holds the shared parts: a curated + local catalog, id->file resolution (with HF download),
a managed directory materialiser, native alias naming, and the single `supports_lora()` gate.

The request layer only passes a LoRA *id* (discovery id, local stem, or HF repo id) plus a
weight, never a raw path, so a client cannot make the backend read an arbitrary file. Resolution
validates the id against the catalog / local dir / HF hub before loading.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback
from utils.paths.storage_roots import studio_root

from .diffusion_families import DIFFUSION_CANCELLED_MSG

# Accepted LoRA formats. safetensors + gguf only (.pt is pickled -> excluded for safety).
_NATIVE_EXTS = (".safetensors", ".gguf")
_DIFFUSERS_EXTS = (".safetensors",)
_ALL_EXTS = (".safetensors", ".gguf")


@dataclass(frozen = True)
class LoraCatalogEntry:
    """One discoverable LoRA adapter."""

    id: str
    display_name: str
    source: str  # "local" | "hub"
    fmt: str  # "safetensors" | "gguf"
    # Compatible family names (empty = unknown, shown but not family-gated). UI greys out
    # incompatible adapters.
    families: tuple[str, ...] = ()
    repo_id: Optional[str] = None  # source == "hub"
    weight_name: Optional[str] = None  # file within the repo (hub)
    local_path: Optional[str] = None  # source == "local"
    size_bytes: int = 0
    weight_default: float = 1.0


@dataclass(frozen = True)
class ResolvedLora:
    """A LoRA resolved to a concrete local file, ready to apply."""

    id: str
    alias: str  # sanitized stem for the <lora:ALIAS:w> tag / diffusers adapter name
    path: str
    fmt: str
    weight: float


# Curated, family-tagged catalog of known-good diffusion LoRAs (HF repos with a single-file
# weight). Local discovery and any public HF LoRA repo id also work.


def _krea2_lora(style: str, display_name: str) -> LoraCatalogEntry:
    """One official krea/Krea-2-LoRA-* style adapter (single ``{style}.safetensors``, trained on
    Krea-2-Raw for Krea-2-Turbo per Krea's guidance)."""
    return LoraCatalogEntry(
        id = f"krea/Krea-2-LoRA-{style}",
        display_name = display_name,
        source = "hub",
        fmt = "safetensors",
        families = ("krea-2",),
        repo_id = f"krea/Krea-2-LoRA-{style}",
        weight_name = f"{style}.safetensors",
    )


_CURATED: tuple[LoraCatalogEntry, ...] = (
    _krea2_lora("retroanime", "Krea 2 Retro Anime"),
    _krea2_lora("neondrip", "Krea 2 Neon Drip"),
    _krea2_lora("darkbrush", "Krea 2 Dark Brush"),
    _krea2_lora("softwatercolor", "Krea 2 Soft Watercolor"),
    _krea2_lora("dotmatrix", "Krea 2 Dot Matrix"),
    _krea2_lora("rainywindow", "Krea 2 Rainy Window"),
    _krea2_lora("vintagetarot", "Krea 2 Vintage Tarot"),
    _krea2_lora("sunsetblur", "Krea 2 Sunset Blur"),
    _krea2_lora("kidsdrawing", "Krea 2 Kids Drawing"),
)


def loras_dir() -> Path:
    """Local directory Studio scans for user-provided diffusion LoRA files."""
    d = studio_root() / "loras" / "diffusion"
    d.mkdir(parents = True, exist_ok = True)
    return d


def sanitize_alias(raw: str) -> str:
    """Deterministic, filesystem- and prompt-tag-safe alias from an id/stem.

    The `<lora:NAME:w>` tag resolves NAME as a filename stem (no path separators, spaces, colons,
    or angle brackets), and the diffusers PEFT adapter name also forbids "." (a module separator),
    so dots are replaced too. The caller breaks cross-source collisions with a numeric suffix.
    """
    stem = raw.rsplit("/", 1)[-1]
    for ext in _ALL_EXTS:
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_-")
    return stem or "lora"


def _scan_local() -> list[LoraCatalogEntry]:
    root = loras_dir()
    try:
        children = sorted(root.iterdir())
    except OSError:
        return []
    files = [p for p in children if p.is_file() and p.suffix.lower() in _ALL_EXTS]
    # Two files sharing a stem but differing in extension collide on id (== stem), so a colliding
    # stem keeps the full filename as its id; a unique stem stays the clean stem.
    stem_counts: dict[str, int] = {}
    for p in files:
        stem_counts[p.stem] = stem_counts.get(p.stem, 0) + 1
    entries: list[LoraCatalogEntry] = []
    for p in files:
        ext = p.suffix.lower()
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        entry_id = p.name if stem_counts.get(p.stem, 0) > 1 else p.stem
        # A ``<stem>.json`` sidecar (written by the trainer on publish) records the adapter's
        # family + default weight so it is family-gated instead of "unknown". Best-effort: a
        # missing/bad sidecar leaves the defaults.
        families, weight_default = _read_lora_sidecar(p)
        entries.append(
            LoraCatalogEntry(
                id = entry_id,
                display_name = entry_id,
                source = "local",
                fmt = "gguf" if ext == ".gguf" else "safetensors",
                local_path = str(p),
                size_bytes = size,
                families = families,
                weight_default = weight_default,
            )
        )
    return entries


def _read_lora_sidecar(weight_path: Path) -> tuple[tuple[str, ...], float]:
    """Read the ``<stem>.json`` sidecar next to a local adapter -> ``(families, weight_default)``.
    Returns ``((), 1.0)`` when absent or unreadable, so discovery never fails on a bad file."""
    sidecar = weight_path.with_suffix(".json")
    try:
        data = json.loads(sidecar.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return (), 1.0
    if not isinstance(data, dict):
        return (), 1.0
    raw_family = data.get("family")
    raw_families = data.get("families")
    names: list[str] = []
    if isinstance(raw_families, (list, tuple)):
        names = [str(f).strip().lower() for f in raw_families if str(f).strip()]
    elif isinstance(raw_family, str) and raw_family.strip():
        names = [raw_family.strip().lower()]
    weight_default = 1.0
    raw_weight = data.get("weight_default")
    if isinstance(raw_weight, (int, float)) and raw_weight > 0:
        weight_default = float(raw_weight)
    return tuple(names), weight_default


def list_loras(*, family: Optional[str] = None) -> list[LoraCatalogEntry]:
    """The merged catalog (curated + local), optionally family-filtered.

    Cheap: one directory scan plus the in-memory curated list. Network is only touched on
    resolve(), when a hub adapter is selected.
    """
    merged = list(_CURATED) + _scan_local()
    if family:
        fam = family.strip().lower()
        merged = [e for e in merged if not e.families or fam in {f.lower() for f in e.families}]
    # Stable order: local first, then by display name.
    merged.sort(key = lambda e: (e.source != "local", e.display_name.lower()))
    return merged


def _catalog_by_id() -> dict[str, LoraCatalogEntry]:
    return {e.id: e for e in (list(_CURATED) + _scan_local())}


def resolve_one(
    spec_id: str,
    weight: float,
    *,
    hf_token: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
) -> ResolvedLora:
    """Resolve a request LoRA id + weight to a concrete local file.

    Accepts a catalog/local id or a bare HF repo id (``owner/name[:weight_file.safetensors]``).
    Downloads hub weights via the xet-fallback helper. Raises FileNotFoundError/ValueError on an
    unresolvable/unsupported id, which the caller maps to a 400.
    """
    # An empty/whitespace token triggers an auth error instead of anonymous access; normalise to None.
    hf_token = hf_token.strip() if hf_token and hf_token.strip() else None
    entry = _catalog_by_id().get(spec_id)
    if entry is not None:
        if entry.source == "local":
            path = entry.local_path or ""
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"LoRA '{spec_id}' is no longer present on disk")
            return ResolvedLora(spec_id, sanitize_alias(spec_id), path, entry.fmt, weight)
        # hub catalog entry
        if not entry.repo_id or not entry.weight_name:
            raise ValueError(f"LoRA '{spec_id}' has no downloadable weight")
        path = hf_hub_download_with_xet_fallback(
            entry.repo_id, entry.weight_name, hf_token, cancel_event = cancel_event
        )
        return ResolvedLora(spec_id, sanitize_alias(spec_id), path, entry.fmt, weight)

    # Not in the catalog: allow a bare public HF repo id (owner/name[:weight_file]).
    if "/" in spec_id:
        repo_id, _, weight_name = spec_id.partition(":")
        weight_name = weight_name or None
        if weight_name is not None:
            # A client-supplied weight file must stay a plain filename inside the repo: reject
            # traversal / absolute paths so it can't resolve outside the HF cache dir.
            if (
                ".." in weight_name
                or weight_name.startswith(("/", "\\", "~"))
                or "\\" in weight_name
                or os.path.isabs(weight_name)
            ):
                raise ValueError(f"invalid LoRA weight file path '{weight_name}'")
        if weight_name is None:
            weight_name = _pick_repo_weight_file(repo_id, hf_token)
        ext = os.path.splitext(weight_name)[1].lower()
        if ext not in _ALL_EXTS:
            raise ValueError(f"unsupported LoRA file '{weight_name}' (need .safetensors/.gguf)")
        path = hf_hub_download_with_xet_fallback(
            repo_id, weight_name, hf_token, cancel_event = cancel_event
        )
        fmt = "gguf" if ext == ".gguf" else "safetensors"
        return ResolvedLora(spec_id, sanitize_alias(repo_id), path, fmt, weight)

    raise FileNotFoundError(
        f"unknown LoRA '{spec_id}': not a local adapter, catalog entry, or HF repo id"
    )


def _pick_repo_weight_file(repo_id: str, hf_token: Optional[str]) -> str:
    """Pick the single LoRA weight file in an HF repo (prefer safetensors)."""
    from huggingface_hub import HfApi

    files = HfApi(token = hf_token).list_repo_files(repo_id)
    safes = [f for f in files if f.lower().endswith(".safetensors") and "/" not in f]
    if len(safes) == 1:
        return safes[0]
    # Prefer a lora-hinting filename, else the first safetensors, else a gguf.
    for f in safes:
        if "lora" in f.lower():
            return f
    if safes:
        return safes[0]
    ggufs = [f for f in files if f.lower().endswith(".gguf") and "/" not in f]
    if ggufs:
        return ggufs[0]
    raise FileNotFoundError(f"no .safetensors/.gguf LoRA file found in '{repo_id}'")


def _scrub_hub_url(msg: str) -> str:
    """Strip embedded http(s) URLs from a Hub error message before it hits a 400 body."""
    cleaned = re.sub(r"https?://\S+", "", msg)
    # Collapse the whitespace the URL removal leaves behind.
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def resolve_specs(
    specs: list[tuple[str, float]],
    *,
    hf_token: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[ResolvedLora]:
    """Resolve request (id, weight) pairs, dropping zero-weight entries.

    Maps the named not-found/gated Hub errors to a 400 (URL scrubbed); does NOT catch the base
    HfHubHTTPError, so a Hub 5xx stays a 500. A mid-download cancel maps to a 409."""
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    out: list[ResolvedLora] = []
    try:
        for spec_id, weight in specs:
            if weight == 0:
                continue
            out.append(resolve_one(spec_id, weight, hf_token = hf_token, cancel_event = cancel_event))
    except (
        FileNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
        EntryNotFoundError,
        GatedRepoError,
    ) as exc:
        raise ValueError(_scrub_hub_url(str(exc))) from exc
    except RuntimeError as exc:
        if str(exc) == "Cancelled":
            raise RuntimeError(DIFFUSION_CANCELLED_MSG) from exc
        raise
    return out


def materialize_native_dir(resolved: list[ResolvedLora], dest: Path) -> list[ResolvedLora]:
    """Populate ``dest`` with symlinks (copy fallback) to the resolved LoRA files.

    sd-cli resolves ``<lora:ALIAS:w>`` against filenames in ``--lora-model-dir``, so each adapter
    needs a uniquely-named file in this dedicated managed directory. Returns the resolved list with
    aliases updated to the (collision-broken) stems written, so the caller injects matching tags.
    """
    dest.mkdir(parents = True, exist_ok = True)
    used: set[str] = set()
    out: list[ResolvedLora] = []
    for r in resolved:
        alias = r.alias
        n = 1
        while alias in used:
            n += 1
            alias = f"{r.alias}_{n}"
        used.add(alias)
        ext = os.path.splitext(r.path)[1].lower() or (
            ".gguf" if r.fmt == "gguf" else ".safetensors"
        )
        link = dest / f"{alias}{ext}"
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(os.path.realpath(r.path), link)
        except OSError:
            import shutil
            shutil.copy2(r.path, link)
        out.append(ResolvedLora(r.id, alias, str(link), r.fmt, r.weight))
    return out


_TAG_RE = re.compile(r"<lora:([^:>]+):([^>]+)>")


def inject_prompt_tags(prompt: str, resolved: list[ResolvedLora]) -> str:
    """Append `<lora:ALIAS:WEIGHT>` tags for the selected adapters, using the validated weights.

    sd-cli strips these tags before the model, so appending is safe. The validated weight (0-2)
    must WIN over any user-typed `<lora:ALIAS:...>`, so strip ALL user tags first (unselected ones
    are dead anyway, not in the managed dir) then append the validated ones.
    """
    # Drop every user-typed tag: unselected ones are dead, selected ones must not override the
    # validated weight / 0-2 bounds.
    cleaned = _TAG_RE.sub("", prompt)
    # Collapse whitespace left by stripped tags.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    tags = [f"<lora:{r.alias}:{_fmt_weight(r.weight)}>" for r in resolved]
    if not tags:
        return cleaned
    sep = "" if not cleaned or cleaned.endswith(" ") else " "
    return f"{cleaned}{sep}{' '.join(tags)}"


def _fmt_weight(w: float) -> str:
    # Stable, compact float formatting (no trailing zeros): 1.0 -> "1", 0.75 -> "0.75".
    s = f"{w:.4f}".rstrip("0").rstrip(".")
    return s or "0"


# Families sd-cli's LoRA name-conversion supports (Qwen-Image has no branch -> excluded).
# Matched by substring against the resolved family name.
_NATIVE_LORA_FAMILY_TOKENS = (
    "flux.1",
    "flux.2",
    "z-image",
    "sd1",
    "sd2",
    "sdxl",
    "sd3",
    "stable-diffusion",
)
# Diffusers quant schemes whose LoRA path is the load-time BAKE (adapters attach on the
# dense transformer BEFORE torchao quantize_ + compile; peft's post-quant TorchaoLoraLinear
# dispatch needs quantizer metadata a manual quantize_ never has). Verified on the Studio
# stack (peft 0.18.1 / torchao 0.17 / torch 2.10): adapter-first, quantize-base-second is
# clean for both schemes -- scale 0 reproduces the quantized base bit-exactly and the wrapped
# transformer compiles.
_DIFFUSERS_LORA_BAKED_QUANT = ("int8", "fp8")
# Prototype schemes with no validated LoRA path (and no shipped families needing one).
_DIFFUSERS_LORA_BLOCKED_QUANT = ("nvfp4", "mxfp8")


def supports_lora(
    *,
    engine: Optional[str],
    family: Optional[str],
    model_kind: Optional[str],
    transformer_quant: Optional[str],
    compiled: bool = False,
) -> bool:
    """Single gate for whether the current load can apply LoRA (status + backends).

    Native (sd_cpp): GGUF via sd-cli, LoRA-capable families only (Qwen excluded). Diffusers:
    bf16 / bnb-4bit apply at generation time (but NOT once the transformer is torch.compile'd:
    diffusers needs the adapter loaded before compilation); torchao int8/fp8 apply via the
    load-time bake (select adapters when loading; a different selection needs a reload), so
    ``compiled`` does not gate them -- the bake precedes compilation by construction. The quant
    check runs BEFORE the gguf-kind check because the quant fast path keeps the PICKER kind
    ("gguf") while the effective transformer is a dense torchao build. nvfp4/mxfp8 stay
    unsupported; GGUF-via-diffusers stays on the native engine for LoRA.
    """
    fam = (family or "").lower()
    if engine == "sd_cpp":
        return any(tok in fam for tok in _NATIVE_LORA_FAMILY_TOKENS)
    # diffusers
    quant = (transformer_quant or "").lower()
    if quant in _DIFFUSERS_LORA_BAKED_QUANT:
        return True  # load-time bake; adapters ride inside the compiled quantized build
    if quant in _DIFFUSERS_LORA_BLOCKED_QUANT:
        return False
    if model_kind == "gguf":
        return False  # GGUF diffusers transformer: use the native engine for LoRA
    if compiled:
        return False  # can't load an adapter onto an already-compiled transformer
    return True
