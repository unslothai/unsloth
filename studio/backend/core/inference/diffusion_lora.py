# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared LoRA support for the Unsloth diffusion backends.

The native sd-cli engine selects adapters by `<lora:NAME:WEIGHT>` prompt tags resolved against a
`--lora-model-dir`; diffusers loads them with `load_lora_weights()` + `set_adapters()`. This
module holds the shared parts: a curated + local catalog, id->file resolution (with HF download),
a managed directory materialiser, native alias naming, and the single `supports_lora()` gate.

The request layer only passes a LoRA *id* (discovery id, local stem, or HF repo id) plus a
weight, never a raw path, so a client cannot make the backend read an arbitrary file. Resolution
validates the id against the catalog / local dir / HF hub before loading.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback
from utils.paths.storage_roots import account_path

from .diffusion_families import DIFFUSION_CANCELLED_MSG
from utils.paths.path_utils import is_appledouble_metadata

# Accepted LoRA formats. safetensors + gguf only (.pt is pickled -> excluded for safety).
_NATIVE_EXTS = (".safetensors", ".gguf")
_DIFFUSERS_EXTS = (".safetensors",)
_ALL_EXTS = (".safetensors", ".gguf")
# ``kind`` in a ``<stem>.json`` sidecar marking the weight beside it as an image LoRA, not model weights.
LORA_SIDECAR_KIND = "diffusion-lora"
_MAX_SCAN_FOLDER_SUBDIRS = 200
_EXPORT_LOCK = threading.Lock()
# ``<stem>.json`` with these stems marks a model or pipeline folder to the model scanners.
_MODEL_SENTINEL_STEMS = frozenset(
    {"config", "adapter_config", "model_index", "modular_model_index"}
)


@dataclass(frozen = True)
class LoraCatalogEntry:
    id: str
    display_name: str
    source: str
    fmt: str
    # Compatible family names (empty = unknown, shown but not family-gated).
    families: tuple[str, ...] = ()
    repo_id: Optional[str] = None
    weight_name: Optional[str] = None  # file within the repo (hub)
    local_path: Optional[str] = None
    size_bytes: int = 0
    weight_default: float = 1.0
    fine_tuned: bool = False


@dataclass(frozen = True)
class ResolvedLora:
    """A LoRA resolved to a concrete local file, ready to apply."""

    id: str
    alias: str  # sanitized stem for the <lora:ALIAS:w> tag / diffusers adapter name
    path: str
    fmt: str
    weight: float


# Curated, family-tagged catalog of known-good diffusion LoRAs (HF repos with a single-file weight). Local discovery and
# any public HF LoRA repo id also work.
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
    d = account_path("loras/diffusion")
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


def _weight_files(root: Path) -> list[Path]:
    try:
        children = sorted(root.iterdir())
    except OSError:
        return []
    return [
        p
        for p in children
        if p.suffix.lower() in _ALL_EXTS and not is_appledouble_metadata(p) and _is_file(p)
    ]


def _is_file(p: Path) -> bool:
    # Path.is_file() re-raises EACCES before Python 3.14; one unreadable entry must not fail the catalog.
    try:
        return p.is_file()
    except OSError:
        return False


def _scan_folder_roots() -> list[Path]:
    """Registered custom model folders plus their direct sub-folders (where an export lands)."""
    try:
        from storage.studio_db import list_scan_folders
        folders = list_scan_folders()
    except Exception:  # noqa: BLE001 -- discovery never fails on the scan-folder table
        return []
    roots: list[Path] = []
    for folder in folders:
        root = Path(folder.get("path") or "")
        try:
            if not root.is_dir():
                continue
            subdirs = sorted(c for c in root.iterdir() if c.is_dir() and not c.name.startswith("."))
        except OSError:
            continue
        roots.append(root)
        roots.extend(subdirs[:_MAX_SCAN_FOLDER_SUBDIRS])
    return roots


def _account_allows():
    """Managed accounts only see adapters (and sidecars) resolving inside paths they may read, so a
    symlink cannot pull another account's or the host's file into listing, generation or export."""
    from hub.services.models import account_access

    if not account_access.managed_account():
        return lambda p: True

    def allows(p: Path) -> bool:
        sidecar = p.with_suffix(".json")
        return account_access.model_visible(str(p)) and (
            not os.path.lexists(sidecar) or account_access.model_visible(str(sidecar))
        )

    return allows


def _open_pinned(path: Path):
    """Open ``path`` and prove the handle is the file now at its resolved, account-readable location,
    so swapping a link between the catalog scan and the copy cannot redirect the read."""
    f = open(path, "rb")
    try:
        real = os.path.realpath(path)
        st, now = os.fstat(f.fileno()), os.stat(real)
        if (st.st_dev, st.st_ino) != (now.st_dev, now.st_ino) or not _account_allows()(Path(real)):
            raise FileNotFoundError(f"LoRA file '{path.name}' is not readable here")
    except BaseException:
        f.close()
        raise
    return f


def _pinned_sidecar(weight_path: Path) -> Optional[dict]:
    try:
        with _open_pinned(weight_path.with_suffix(".json")) as f:
            data = json.loads(f.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _scan_local() -> list[LoraCatalogEntry]:
    root_dir = loras_dir()
    allows = _account_allows()
    files = [p for p in _weight_files(root_dir) if allows(p)]
    # Two files sharing a stem but differing in extension collide on id (== stem), so a colliding stem keeps the full
    # filename.
    stem_counts: dict[str, int] = {}
    for p in files:
        stem_counts[p.stem] = stem_counts.get(p.stem, 0) + 1
    found = [(p, p.name if stem_counts.get(p.stem, 0) > 1 else p.stem) for p in files]
    used = {entry_id for _, entry_id in found}
    seen = {os.path.normcase(os.path.realpath(p)) for p in files}
    # Custom models folders contribute only sidecar-marked image LoRAs, so model weights never show up here.
    for root in _scan_folder_roots():
        for p in _weight_files(root):
            key = os.path.normcase(os.path.realpath(p))
            if key in seen or not is_image_lora_file(p) or not allows(p):
                continue
            seen.add(key)
            # Keyed on the file's own path, never on scan order: saved recipes must not drift onto
            # another folder's same-named adapter when folders are added or removed.
            entry_id = f"{p.stem}-{hashlib.sha1(os.fsencode(key)).hexdigest()[:8]}"
            if entry_id in used:
                continue
            used.add(entry_id)
            found.append((p, entry_id))

    entries: list[LoraCatalogEntry] = []
    for p, entry_id in found:
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        # A ``<stem>.json`` sidecar (written by the trainer on publish) records the adapter's family + default weight so
        # it is family-gated instead of "unknown". Best-effort.
        families, weight_default = _read_lora_sidecar(p)
        entries.append(
            LoraCatalogEntry(
                id = entry_id,
                display_name = entry_id if p.parent == root_dir else p.stem,
                source = "local",
                fmt = "gguf" if p.suffix.lower() == ".gguf" else "safetensors",
                local_path = str(p),
                size_bytes = size,
                families = families,
                weight_default = weight_default,
                fine_tuned = (_sidecar_data(p) or {}).get("source") == "studio-trained",
            )
        )
    return entries


def _sidecar_data(weight_path: Path) -> Optional[dict]:
    try:
        data = json.loads(weight_path.with_suffix(".json").read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_image_lora_file(path: Path) -> bool:
    """A .safetensors/.gguf whose sidecar marks it as an image LoRA (``kind``, or a trainer sidecar
    from before the marker). Model scanners use this to keep image LoRAs out of model listings."""
    if path.suffix.lower() not in _ALL_EXTS:
        return False
    data = _sidecar_data(path)
    return data is not None and (
        data.get("kind") == LORA_SIDECAR_KIND or data.get("source") == "studio-trained"
    )


def _read_lora_sidecar(weight_path: Path) -> tuple[tuple[str, ...], float]:
    """Read the ``<stem>.json`` sidecar next to a local adapter -> ``(families, weight_default)``.
    Returns ``((), 1.0)`` when absent or unreadable, so discovery never fails on a bad file."""
    data = _sidecar_data(weight_path)
    if data is None:
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


def _staging_name(dest_dir: Path, stem: str) -> str:
    # Not mkstemp: its 0600 file would publish an owner-only marker; a fresh open() honours the umask.
    import secrets

    # Short and stem-free: a near-255-byte stem plus a token would overflow the name limit.
    return str(dest_dir / f".lora-export.{secrets.token_hex(8)}.part")


def export_local_lora(lora_id: str, dest_dir: Path) -> Path:
    """Copy a local adapter into ``dest_dir`` with a ``<stem>.json`` sidecar carrying the image LoRA marker.

    Takes a catalog id, never a path, so only listed image LoRAs can be read. Returns the copied weight
    file; a stem already used by other bytes, another weight format or a foreign ``.json`` gets a suffix.
    """
    import filecmp
    import shutil

    entry = next((e for e in _scan_local() if e.id == lora_id), None)
    if entry is None or not entry.local_path:
        raise FileNotFoundError(f"no local image LoRA named '{lora_id}'")
    src = Path(entry.local_path)
    dest_dir.mkdir(parents = True, exist_ok = True)

    def _free(out: Path) -> bool:
        if out.stem.lower() in _MODEL_SENTINEL_STEMS:
            return False
        if out.exists() and os.path.samefile(src, out):
            return True
        # The sidecar is per stem, so a sibling weight of another format would share it (case-insensitive FS too).
        if any(
            c.name != out.name
            and c.stem.casefold() == out.stem.casefold()
            and c.suffix.lower() in _ALL_EXTS
            for c in dest_dir.iterdir()
        ):
            return False
        if not out.exists():
            return not out.with_suffix(".json").exists()
        return filecmp.cmp(src, out, shallow = False) and (
            not out.with_suffix(".json").exists() or is_image_lora_file(out)
        )

    with _EXPORT_LOCK:
        out, n = dest_dir / src.name, 2
        while not _free(out):
            out = dest_dir / f"{src.stem}-{n}{src.suffix}"
            n += 1
        meta = json.dumps({**(_pinned_sidecar(src) or {}), "kind": LORA_SIDECAR_KIND}, indent = 2)
        # Exporting into the folder it already sits in would copy a file onto itself.
        copy = not (out.exists() and os.path.samefile(src, out))
        # Both files are staged under names no scanner reads, so a failed export leaves nothing behind.
        sidecar = out.with_suffix(".json")
        staged: list[str] = []
        new_sidecar = not sidecar.exists()
        try:
            if copy:
                staged.append(_staging_name(dest_dir, src.stem))
                with _open_pinned(src) as fsrc, open(staged[-1], "xb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
                    st = os.fstat(fsrc.fileno())
                os.chmod(staged[-1], stat.S_IMODE(st.st_mode))
                os.utime(staged[-1], ns = (st.st_atime_ns, st.st_mtime_ns))
            staged.append(_staging_name(dest_dir, src.stem))
            Path(staged[-1]).write_text(meta, encoding = "utf-8")
            # Marker first: a scanner must never see the weight unmarked.
            os.replace(staged[-1], sidecar)
            if copy:
                os.replace(staged[0], out)
        except BaseException:
            for tmp in staged:
                Path(tmp).unlink(missing_ok = True)
            if new_sidecar and not out.exists():
                sidecar.unlink(missing_ok = True)
            raise
    return out


def resolve_one(
    spec_id: str,
    weight: float,
    *,
    family: Optional[str] = None,
    hf_token: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
    catalog: Optional[dict[str, LoraCatalogEntry]] = None,
) -> ResolvedLora:
    """Resolve a request LoRA id + weight to a concrete local file.

    Accepts a catalog/local id or a bare HF repo id (``owner/name[:weight_file.safetensors]``).
    Downloads hub weights via the xet-fallback helper. Raises FileNotFoundError/ValueError on an
    unresolvable/unsupported id, which the caller maps to a 400.

    ``family`` (the loaded model family) enforces catalog family tags HERE, not only in the picker,
    so a direct API client cannot load a LoRA tagged for another family through the wrong pipeline.
    An untagged catalog entry (empty ``families``) stays unrestricted.
    """
    # An empty token triggers an auth error instead of anonymous access; normalise to None.
    hf_token = hf_token.strip() if hf_token and hf_token.strip() else None
    entry = (_catalog_by_id() if catalog is None else catalog).get(spec_id)
    if entry is not None:
        req_fam = (family or "").strip().lower()
        if entry.families and req_fam and req_fam not in {f.lower() for f in entry.families}:
            raise ValueError(
                f"LoRA '{spec_id}' is for {', '.join(entry.families)}, not the loaded "
                f"'{family}' model family; pick a LoRA built for this family."
            )
        if entry.source == "local":
            path = entry.local_path or ""
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"LoRA '{spec_id}' is no longer present on disk")
            # Re-checked here: a catalog can be built well before an earlier stacked LoRA finishes downloading.
            if not _account_allows()(Path(path)):
                raise FileNotFoundError(f"LoRA '{spec_id}' is no longer present on disk")
            return ResolvedLora(spec_id, sanitize_alias(spec_id), path, entry.fmt, weight)
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
            # A client-supplied weight file must stay a plain filename inside the repo: reject traversal / absolute
            # paths so it cannot resolve outside the HF cache dir.
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

    from hub.utils.gguf import drop_shadowed_appledouble_names, is_imatrix_filename

    files = drop_shadowed_appledouble_names(list(HfApi(token = hf_token).list_repo_files(repo_id)))
    safes = [f for f in files if f.lower().endswith(".safetensors") and "/" not in f]
    if len(safes) == 1:
        return safes[0]
    # Prefer a lora-hinting filename, else the first safetensors, else a gguf.
    for f in safes:
        if "lora" in f.lower():
            return f
    if safes:
        return safes[0]
    # The gguf fallback only: an imatrix is a .gguf holding no adapter and would be picked here, while a .safetensors
    # is never one, so the candidates above stay untouched.
    ggufs = [
        f
        for f in files
        if f.lower().endswith(".gguf") and "/" not in f and not is_imatrix_filename(f)
    ]
    if ggufs:
        return ggufs[0]
    raise FileNotFoundError(f"no .safetensors/.gguf LoRA file found in '{repo_id}'")


def _scrub_hub_url(msg: str) -> str:
    """Strip embedded http(s) URLs from a Hub error message before it hits a 400 body."""
    cleaned = re.sub(r"https?://\S+", "", msg)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def resolve_specs(
    specs: list[tuple[str, float]],
    *,
    family: Optional[str] = None,
    hf_token: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[ResolvedLora]:
    """Resolve request (id, weight) pairs, dropping zero-weight entries.

    ``family`` (the loaded model family) enforces catalog family tags in :func:`resolve_one` for
    direct API callers. Maps the named not-found/gated Hub errors to a 400 (URL scrubbed); does NOT
    catch the base HfHubHTTPError, so a Hub 5xx stays a 500. A mid-download cancel maps to a 409."""
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    out: list[ResolvedLora] = []
    # One custom-folder scan per request, not one per stacked LoRA.
    catalog = _catalog_by_id() if any(weight != 0 for _, weight in specs) else {}
    try:
        for spec_id, weight in specs:
            if weight == 0:
                continue
            out.append(
                resolve_one(
                    spec_id,
                    weight,
                    family = family,
                    hf_token = hf_token,
                    cancel_event = cancel_event,
                    catalog = catalog,
                )
            )
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
    # drop every user-typed tag: unselected ones are dead, selected ones must not override the validated weight / 0-2
    # bounds
    cleaned = _TAG_RE.sub("", prompt)
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


# Families sd-cli's LoRA name-conversion supports (Qwen-Image has no branch). Matched by substring against the resolved
# family name.
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
# Diffusers quant schemes whose LoRA path is the load-time BAKE: adapters attach on the dense transformer BEFORE
# torchao quantize_ + compile (peft's TorchaoLoraLinear dispatch needs quantizer metadata a manual quantize_ lacks).
# Verified on peft 0.18.1 / torchao 0.17 / torch 2.10: scale 0 reproduces the quantized base bit-exactly.
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
