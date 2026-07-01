"""Shared LoRA support for the Studio diffusion backends.

Both engines apply LoRA differently -- the native stable-diffusion.cpp CLI selects
adapters by `<lora:NAME:WEIGHT>` prompt tags resolved against a `--lora-model-dir`,
while diffusers loads them with `load_lora_weights()` + `set_adapters()`. This module
holds the parts they share: a curated + local catalog, id->file resolution (with HF
download), a managed directory materialiser for the native tier, deterministic native
alias naming, and the single `supports_lora()` gate the UI/status and both backends use.

The request layer only ever passes a LoRA *id* (a discovery id, local stem, or HF repo
id) plus a weight -- never a raw filesystem path -- so a client cannot make the backend
read an arbitrary file. Resolution validates the id against the catalog / the local LoRA
directory / the HF hub before anything is loaded.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback
from utils.paths.storage_roots import studio_root

# LoRA file formats we accept. sd-cli probes .safetensors/.gguf/.pt; diffusers loads
# .safetensors. We expose safetensors + gguf (pt is legacy/pickled -> excluded for safety).
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
    # Family names this LoRA is compatible with (see diffusion_families). Empty = unknown
    # (shown but not family-gated). Used by the UI to grey out incompatible adapters.
    families: tuple[str, ...] = ()
    repo_id: Optional[str] = None  # for source == "hub"
    weight_name: Optional[str] = None  # file within the repo (source == "hub")
    local_path: Optional[str] = None  # for source == "local"
    size_bytes: int = 0
    weight_default: float = 1.0


@dataclass(frozen = True)
class ResolvedLora:
    """A LoRA resolved to a concrete local file, ready to apply."""

    id: str
    alias: str  # sanitized stem used for the native <lora:ALIAS:w> tag / diffusers adapter name
    path: str
    fmt: str
    weight: float


# Curated, family-tagged catalog of known-good diffusion LoRAs. Kept intentionally small
# and data-driven; extend as unsloth hosts/curates more. Entries are HF repos with a
# single-file weight. (Left minimal on purpose -- local discovery is the primary source,
# and users can also reference any public HF LoRA repo id directly.)
_CURATED: tuple[LoraCatalogEntry, ...] = ()


def loras_dir() -> Path:
    """Local directory Studio scans for user-provided diffusion LoRA files."""
    d = studio_root() / "loras" / "diffusion"
    d.mkdir(parents = True, exist_ok = True)
    return d


def sanitize_alias(raw: str) -> str:
    """Deterministic, filesystem- and prompt-tag-safe alias from an id/stem.

    The native `<lora:NAME:w>` tag resolves NAME as a filename stem, so the alias must
    contain no path separators, spaces, colons, or angle brackets. Collisions across
    sources are broken by the caller (materialize_native_dir) with a numeric suffix.
    """
    stem = raw.rsplit("/", 1)[-1]
    for ext in _ALL_EXTS:
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "lora"


def _scan_local() -> list[LoraCatalogEntry]:
    entries: list[LoraCatalogEntry] = []
    root = loras_dir()
    try:
        children = sorted(root.iterdir())
    except OSError:
        return entries
    for p in children:
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in _ALL_EXTS:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        entries.append(
            LoraCatalogEntry(
                id = p.stem,
                display_name = p.stem,
                source = "local",
                fmt = "gguf" if ext == ".gguf" else "safetensors",
                local_path = str(p),
                size_bytes = size,
            )
        )
    return entries


def list_loras(*, family: Optional[str] = None) -> list[LoraCatalogEntry]:
    """Return the merged catalog (curated + local), optionally family-filtered.

    Cheap: a single directory scan plus the in-memory curated list. Network is only
    touched later, on resolve(), when a hub adapter is actually selected.
    """
    merged = list(_CURATED) + _scan_local()
    if family:
        fam = family.strip().lower()
        merged = [e for e in merged if not e.families or fam in {f.lower() for f in e.families}]
    # Stable order: local first (user intent), then by display name.
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

    Accepts, in order: a catalog/local id, or a bare HF repo id (``owner/name`` with an
    optional ``owner/name:weight_file.safetensors`` suffix). Downloads hub weights via the
    shared xet-fallback helper. Raises FileNotFoundError/ValueError on an unresolvable or
    unsupported id -- the caller maps that to a clear 400.
    """
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
    # Prefer a filename hinting at a lora, else the first safetensors, else a gguf.
    for f in safes:
        if "lora" in f.lower():
            return f
    if safes:
        return safes[0]
    ggufs = [f for f in files if f.lower().endswith(".gguf") and "/" not in f]
    if ggufs:
        return ggufs[0]
    raise FileNotFoundError(f"no .safetensors/.gguf LoRA file found in '{repo_id}'")


def resolve_specs(
    specs: list[tuple[str, float]],
    *,
    hf_token: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[ResolvedLora]:
    """Resolve request (id, weight) pairs, dropping zero-weight entries."""
    out: list[ResolvedLora] = []
    for spec_id, weight in specs:
        if weight == 0:
            continue
        out.append(resolve_one(spec_id, weight, hf_token = hf_token, cancel_event = cancel_event))
    return out


def materialize_native_dir(resolved: list[ResolvedLora], dest: Path) -> list[ResolvedLora]:
    """Populate ``dest`` with symlinks (copy fallback) to the resolved LoRA files.

    sd-cli scans ``--lora-model-dir`` and resolves ``<lora:ALIAS:w>`` against filenames in
    it, so each selected adapter needs a uniquely-named file there. We use a dedicated
    managed directory (never a broad cache dir), which keeps the directory scan small and
    safe. Returns the resolved list with aliases updated to the (collision-broken) stems
    actually written, so the caller injects matching prompt tags.
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
    """Append `<lora:ALIAS:WEIGHT>` tags to the prompt, skipping any the user already typed.

    sd-cli strips these tags before they reach the model, so appending them is safe and
    deterministic. Duplicate protection: if the prompt already contains a tag for the same
    alias, we don't add a second one.
    """
    existing = {m.group(1) for m in _TAG_RE.finditer(prompt)}
    tags = [
        f"<lora:{r.alias}:{_fmt_weight(r.weight)}>" for r in resolved if r.alias not in existing
    ]
    if not tags:
        return prompt
    sep = "" if not prompt or prompt.endswith(" ") else " "
    return f"{prompt}{sep}{' '.join(tags)}"


def _fmt_weight(w: float) -> str:
    # Stable, compact float formatting (no trailing zeros): 1.0 -> "1", 0.75 -> "0.75".
    s = f"{w:.4f}".rstrip("0").rstrip(".")
    return s or "0"


# Families the native sd-cli LoRA name-conversion supports (SD1.5/SD2/SDXL/SD3/FLUX/
# z-image). Qwen-Image has no LoRA branch in stable-diffusion.cpp -> excluded until
# validated. Matched by substring against the resolved family name.
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
# Diffusers quant schemes that cannot take LoRA cleanly (torchao tensor-subclass weights).
_DIFFUSERS_LORA_BLOCKED_QUANT = ("int8", "fp8", "nvfp4", "mxfp8")


def supports_lora(
    *,
    engine: Optional[str],
    family: Optional[str],
    model_kind: Optional[str],
    transformer_quant: Optional[str],
) -> bool:
    """Single gate for whether the current load can apply LoRA (used by status + backends).

    Native (sd_cpp): GGUF via sd-cli, for the LoRA-capable families only (Qwen excluded).
    Diffusers: bf16 or bnb-4bit transformers, but NOT the dense torchao fp8/int8 fast path
    (tensor-subclass weights) and NOT GGUF-via-diffusers.
    """
    fam = (family or "").lower()
    if engine == "sd_cpp":
        return any(tok in fam for tok in _NATIVE_LORA_FAMILY_TOKENS)
    # diffusers
    if model_kind == "gguf":
        return False  # GGUF diffusers transformer: use the native engine for LoRA
    if transformer_quant and transformer_quant.lower() in _DIFFUSERS_LORA_BLOCKED_QUANT:
        return False
    return True
