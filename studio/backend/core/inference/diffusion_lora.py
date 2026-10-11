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

# .pt is pickled, excluded for safety.
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
    families: tuple[str, ...] = ()
    repo_id: Optional[str] = None
    weight_name: Optional[str] = None
    local_path: Optional[str] = None
    size_bytes: int = 0
    weight_default: float = 1.0
    fine_tuned: bool = False
    # Id of the loras/diffusion adapter this custom-folder file is an export of: listed once, still resolvable.
    copy_of: Optional[str] = None


@dataclass(frozen = True)
class ResolvedLora:
    """A LoRA resolved to a concrete local file, ready to apply."""

    id: str
    alias: str
    path: str
    fmt: str
    weight: float


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
    """Dots are replaced too, as PEFT forbids them; the tag NAME may not hold spaces or colons."""
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


def _is_dir(p: Path) -> bool:
    try:
        return p.is_dir()
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
        if not _is_dir(root):
            continue
        try:
            children = list(root.iterdir())
        except OSError:
            children = []
        subdirs = sorted(c for c in children if not c.name.startswith(".") and _is_dir(c))
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
            # Keyed on the path, not scan order, so saved recipes never drift to another folder's same-named file.
            entry_id = f"{p.stem}-{hashlib.sha1(os.fsencode(key)).hexdigest()[:8]}"
            if entry_id in used:
                continue
            used.add(entry_id)
            found.append((p, entry_id))

    entries: list[LoraCatalogEntry] = []
    # An export carries its source's sidecar plus ``kind``. Only trainer sidecars (``created_at``) qualify, and the
    # bytes at both ends must match too: equal size and metadata alone fit two different same-rank LoRAs.
    originals: dict[tuple, tuple[str, Path]] = {}
    for p, entry_id in found:
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        sidecar = _sidecar_data(p) or {}
        meta = {k: v for k, v in sidecar.items() if k != "kind"}
        signature = (
            (size, json.dumps(meta, sort_keys = True, default = str)) if "created_at" in meta else None
        )
        copy_of = None
        if p.parent == root_dir:
            if signature:
                originals.setdefault(signature, (entry_id, p))
        elif signature in originals and _same_ends(originals[signature][1], p):
            copy_of = originals[signature][0]
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
                fine_tuned = sidecar.get("source") == "studio-trained",
                copy_of = copy_of,
            )
        )
    return entries


def _same_ends(
    a: Path,
    b: Path,
    n: int = 1 << 16,
) -> bool:
    try:
        with _open_pinned(a) as fa, _open_pinned(b) as fb:
            if fa.read(n) != fb.read(n):
                return False
            for f in (fa, fb):
                f.seek(max(0, os.fstat(f.fileno()).st_size - n))
            return fa.read(n) == fb.read(n)
    except OSError:
        return False


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


def list_loras(
    *, family: Optional[str] = None, include_copies: bool = False
) -> list[LoraCatalogEntry]:
    """The merged catalog (curated + local), optionally family-filtered.

    Cheap: one directory scan plus the in-memory curated list. Network is only touched on
    resolve(), when a hub adapter is selected.
    """
    merged = [e for e in list(_CURATED) + _scan_local() if include_copies or not e.copy_of]
    if family:
        fam = family.strip().lower()
        merged = [e for e in merged if not e.families or fam in {f.lower() for f in e.families}]
    merged.sort(key = lambda e: (e.source != "local", e.display_name.lower()))
    return merged


def _catalog_by_id() -> dict[str, LoraCatalogEntry]:
    return {e.id: e for e in (list(_CURATED) + _scan_local())}


def _staging_name(dest_dir: Path, stem: str) -> str:
    # Not mkstemp: its 0600 file would publish an owner-only marker; a fresh open() honours the umask.
    import secrets
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
    """Enforces family tags here, not only in the picker, so direct API calls cannot skip them."""
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

    if "/" in spec_id:
        repo_id, _, weight_name = spec_id.partition(":")
        weight_name = weight_name or None
        if weight_name is not None:
            # Reject traversal / absolute paths so the file stays inside the HF cache.
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
    for f in safes:
        if "lora" in f.lower():
            return f
    if safes:
        return safes[0]
    # gguf fallback only: skip imatrix files, which hold no adapter.
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
    """Maps not-found and gated Hub errors to 400 but not base HfHubHTTPError, so a Hub 5xx stays a 500."""
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
    """Strips all user <lora:...> tags first so the validated weight wins over any typed weight."""
    cleaned = _TAG_RE.sub("", prompt)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    tags = [f"<lora:{r.alias}:{_fmt_weight(r.weight)}>" for r in resolved]
    if not tags:
        return cleaned
    sep = "" if not cleaned or cleaned.endswith(" ") else " "
    return f"{cleaned}{sep}{' '.join(tags)}"


def _fmt_weight(w: float) -> str:
    s = f"{w:.4f}".rstrip("0").rstrip(".")
    return s or "0"


# Families sd-cli LoRA name conversion supports (not Qwen-Image); substring match.
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
# Adapters bake on the dense transformer BEFORE torchao quantize_ + compile: peft's
# TorchaoLoraLinear needs quantizer metadata a manual quantize_ lacks.
_DIFFUSERS_LORA_BAKED_QUANT = ("int8", "fp8")
_DIFFUSERS_LORA_BLOCKED_QUANT = ("nvfp4", "mxfp8")


def supports_lora(
    *,
    engine: Optional[str],
    family: Optional[str],
    model_kind: Optional[str],
    transformer_quant: Optional[str],
    compiled: bool = False,
) -> bool:
    """Adapters load before torch.compile; torchao int8/fp8 bake at load, so compile does not gate them."""
    fam = (family or "").lower()
    if engine == "sd_cpp":
        return any(tok in fam for tok in _NATIVE_LORA_FAMILY_TOKENS)
    quant = (transformer_quant or "").lower()
    if quant in _DIFFUSERS_LORA_BAKED_QUANT:
        return True
    if quant in _DIFFUSERS_LORA_BLOCKED_QUANT:
        return False
    if model_kind == "gguf":
        return False
    if compiled:
        return False
    return True
