# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Component sizes of a diffusers pipeline from safetensors headers alone (no weight byte read).

One header parser for every size helper, one ``ComponentSizes`` for image and video, read from a local snapshot or,
before anything is downloaded, from the Hub through ranged header reads. The hand-written family tables
(``VideoFamily.bf16_components_gb``, ``_FAMILY_BF16_GB``) stay as the offline fallback: ``video_components_gb``
reconciles the two and records every disagreement instead of silently switching numbers.
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# Default cap of the size helpers; callers that historically used a tighter cap pass their own.
MAX_HEADER_BYTES = 256 * 1024 * 1024
# Every floating dtype safetensors names, with its stored width. Float8 (and the sub-byte float4 / MX scale types) are
# floating parameters: a float8-stored checkpoint loaded at bf16 / fp16 occupies 2 bytes per parameter on device.
FLOAT_WIDTH = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2FNUZ": 1,
    "F8_E8M0": 1,
}
_GB = 1000.0**3

ReadRange = Callable[[int, int], bytes]


def parse_safetensors_header(
    read: Callable[[int], bytes], *, max_bytes: int = MAX_HEADER_BYTES
) -> dict:
    """The JSON header (``__metadata__`` included) given a sequential ``read(n)``. Raises ValueError on a file that is
    not safetensors or whose header length is implausible."""
    raw = read(8)
    if len(raw) != 8:
        raise ValueError("not a safetensors file")
    (length,) = struct.unpack("<Q", raw)
    if not 0 < length <= max_bytes:
        raise ValueError("implausible safetensors header length")
    body = read(length)
    if len(body) != length:
        raise ValueError("truncated safetensors header")
    header = json.loads(body)
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not an object")
    return header


def read_safetensors_header(path: Any, *, max_bytes: int = MAX_HEADER_BYTES) -> Optional[dict]:
    """``parse_safetensors_header`` of a local file, or None when unreadable. Never raises."""
    try:
        with open(path, "rb") as fh:
            return parse_safetensors_header(fh.read, max_bytes=max_bytes)
    except Exception:  # noqa: BLE001 - unreadable / crafted header: no claim about this file
        return None


def tensor_entries(header: Optional[dict]) -> Iterable[tuple[str, dict]]:
    """``(name, meta)`` for every tensor entry (``__metadata__`` and malformed rows skipped)."""
    for name, meta in (header or {}).items():
        if name != "__metadata__" and isinstance(meta, dict):
            yield name, meta


def entry_numel(meta: dict) -> int:
    numel = 1
    for dim in meta.get("shape", []) or []:
        numel *= int(dim)
    return numel


def entry_nbytes(meta: dict) -> int:
    begin, end = meta["data_offsets"]
    return int(end) - int(begin)


@dataclass(frozen=True)
class ComponentSize:
    """One pipeline component: stored bytes, and the parameter split the cast arithmetic needs."""

    name: str
    stored_bytes: int
    float_params: int
    nonfloat_bytes: int
    # stored bytes per float dtype ("BF16", "F32", ...)
    bytes_by_dtype: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    files: tuple[str, ...] = field(default_factory=tuple)

    @property
    def params(self) -> int:
        return self.float_params

    def resident_bytes(self, itemsize: int = 2) -> int:
        """Bytes on device after every float tensor is cast to ``itemsize`` bytes (non-float tensors as stored)."""
        return self.float_params * int(itemsize) + self.nonfloat_bytes

    def resident_gb(self, itemsize: int = 2) -> float:
        return self.resident_bytes(itemsize) / _GB


@dataclass(frozen=True)
class ComponentSizes:
    """Per-component sizes of one checkpoint (``source`` = "local" or "hub", ``revision`` when known)."""

    components: dict
    source: str = "local"
    revision: Optional[str] = None

    def get(self, name: str) -> Optional[ComponentSize]:
        return self.components.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.components))

    def resident_gb(self, names: Iterable[str], itemsize: int = 2) -> float:
        return (
            sum(self.components[n].resident_bytes(itemsize) for n in names if n in self.components)
            / _GB
        )


def _accumulate(
    name: str, headers: Iterable[tuple[str, Optional[dict]]]
) -> Optional[ComponentSize]:
    stored = float_params = nonfloat = 0
    by_dtype: dict[str, int] = {}
    files: list[str] = []
    seen: set = set()
    for fname, header in headers:
        if header is None:
            return None  # a missing shard makes the component unknown, never smaller
        files.append(fname)
        for tname, meta in tensor_entries(header):
            if tname in seen:
                continue
            seen.add(tname)
            try:
                nbytes = entry_nbytes(meta)
                dtype = str(meta.get("dtype", ""))
                numel = entry_numel(meta)
            except Exception:  # noqa: BLE001
                return None
            stored += nbytes
            if dtype in FLOAT_WIDTH:
                float_params += numel
                by_dtype[dtype] = by_dtype.get(dtype, 0) + nbytes
            else:
                nonfloat += nbytes
    if not files or stored <= 0:
        return None
    return ComponentSize(
        name=name,
        stored_bytes=stored,
        float_params=float_params,
        nonfloat_bytes=nonfloat,
        bytes_by_dtype=tuple(sorted(by_dtype.items())),
        files=tuple(files),
    )


# A default-variant weight file of a diffusers / transformers component (no ``.fp16`` style variant infix).
_DEFAULT_WEIGHT_RE = re.compile(
    r"^(?:diffusion_pytorch_model|model)(?:-\d{5}-of-\d{5})?\.safetensors$"
)
_SHARD_RE = re.compile(r"^.+-\d{5}-of-\d{5}\.safetensors$")
_INDEX_RE = re.compile(r"^(?:diffusion_pytorch_model|model)\.safetensors\.index\.json$")


_TRANSFORMERS_FOLDER_RE = re.compile(r"^(?:text_encoder|image_encoder|prompt_enhancer)(?:_\d+)?$")


def _stem_order(folder: str) -> tuple[str, str]:
    """Weight-file stems in load priority: a transformers component reads ``model.*`` first, a diffusers one
    ``diffusion_pytorch_model.*`` (LTX-2's text_encoder/ ships both, and only one is what loads)."""
    if _TRANSFORMERS_FOLDER_RE.match(folder or ""):
        return ("model", "diffusion_pytorch_model")
    return ("diffusion_pytorch_model", "model")


def select_weight_files(
    names: Iterable[str], index_files: Optional[Iterable[str]] = None, folder: str = ""
) -> list[str]:
    """The safetensors files ``from_pretrained`` reads in one component folder (default variant). ``index_files``
    are the shard names an index lists; when given they win (a stray shard of an older layout is not counted)."""
    names = [n for n in names if n.endswith(".safetensors")]
    if index_files:
        listed = sorted(set(index_files))
        if all(n in names for n in listed):
            return listed
        return []  # an index naming a file not present: incomplete, unknown
    # Without an index only an UNSHARDED file is a complete answer: shards alone (index missing or not yet downloaded)
    # make the component unknown, never the sum of whichever shards happen to be here.
    for stem in _stem_order(folder):
        if f"{stem}.safetensors" in names:
            return [f"{stem}.safetensors"]
        if any(_SHARD_RE.match(n) and n.startswith(stem + "-") for n in names):
            return []
    if any(_SHARD_RE.match(n) for n in names):
        return []
    plain = sorted(n for n in names if "." not in n[: -len(".safetensors")])
    return plain


def _index_shards(text: str) -> Optional[list[str]]:
    try:
        weight_map = json.loads(text).get("weight_map") or {}
        return sorted(set(str(v) for v in weight_map.values()))
    except Exception:  # noqa: BLE001
        return None


def _pick_files(
    names: Iterable[str], read_text: Callable[[str], str], folder: str = ""
) -> list[str]:
    """Weight files of one folder. Only the loader's PREFERRED index counts (``model`` first for a transformers
    folder, ``diffusion_pytorch_model`` first for a diffusers one): a folder can carry a stale second index, and the
    loader never falls back to it. That index unreadable, or naming a shard that is not here (a partial download),
    makes the component unknown, never the smaller present subset nor a complete lower-priority index."""
    names = list(names)
    for stem in _stem_order(folder):
        index = f"{stem}.safetensors.index.json"
        if index in names:
            try:
                shards = _index_shards(read_text(index))
            except Exception:  # noqa: BLE001
                shards = None
            if shards and all(s in names for s in shards):
                return select_weight_files(names, shards, folder)
            return []
        if any(
            _DEFAULT_WEIGHT_RE.match(n) and n.startswith((stem + ".", stem + "-")) for n in names
        ):
            # this stem's weights without an index: the loader stops here too (shards alone are unknown)
            return select_weight_files(
                [n for n in names if n.startswith((stem + ".", stem + "-"))], folder=folder
            )
    return select_weight_files(names, folder=folder)


def local_component_sizes(
    root: Any, components: Optional[Iterable[str]] = None
) -> Optional[ComponentSizes]:
    """Component sizes of a local diffusers directory (every top-level folder holding safetensors, or just
    ``components``). None when the directory is unreadable or holds no weights. Never raises."""
    try:
        base = Path(root)
        if not base.is_dir():
            return None
        wanted = set(components) if components is not None else None
        out: dict[str, ComponentSize] = {}
        for sub in sorted(p for p in base.iterdir() if p.is_dir()):
            if wanted is not None and sub.name not in wanted:
                continue
            names = [p.name for p in sub.iterdir() if p.is_file()]
            files = _pick_files(
                names, lambda n, sub=sub: (sub / n).read_text(encoding="utf-8"), sub.name
            )
            if not files:
                continue
            size = _accumulate(sub.name, ((f, read_safetensors_header(sub / f)) for f in files))
            if size is not None:
                out[sub.name] = size
        return ComponentSizes(out, source="local") if out else None
    except Exception:  # noqa: BLE001
        return None


def active_cache_dir() -> Optional[str]:
    """Studio's LIVE hub cache root (the one every loader is pinned to), not huggingface_hub's import-time default."""
    try:
        from utils.hf_cache_settings import active_hf_hub_cache

        return active_hf_hub_cache()
    except Exception:  # noqa: BLE001
        return None


def local_snapshot_dir(repo_id: str, cache_dir: Any = None) -> Optional[Path]:
    """The newest cached snapshot of ``repo_id`` (None when not cached); no network."""
    try:
        from .diffusion_small_host import resolve_snapshot_dir

        return resolve_snapshot_dir(repo_id, cache_dir if cache_dir is not None else active_cache_dir())
    except Exception:  # noqa: BLE001
        return None


# ---- Hub (ranged header reads, before download) ----

HUB_TIMEOUT_SECONDS = 20.0
HUB_RETRY_SECONDS = 600.0
# A size read through a mutable ref (None / "main" / a branch / a tag) is revalidated after this long: one metadata call
# resolves the ref, and header reads only repeat when it now points at a different commit.
HUB_REFRESH_SECONDS = 600.0
_HUB_CACHE: dict = {}
# (repo, resolved commit sha, components, token id) -> sizes: immutable, so never re-read.
_HUB_SHA_CACHE: dict = {}
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _token_id(token: Any) -> Optional[str]:
    """Cache partition for a token (a digest, never the token): a gated repo read with one token is not served to a
    load without it."""
    if not token or not isinstance(token, str):
        return None if not token else "implicit"
    import hashlib

    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


_HUB_LOCK = threading.Lock()


def hub_disabled() -> bool:
    """Ranged Hub reads are off when the Hub is offline, inside a local-files-only load (the prequant scope's
    ``_LOCAL_FILES_ONLY``), or with ``UNSLOTH_DIFFUSION_HEADER_FETCH=0``."""
    try:
        from .diffusion_prequant import _LOCAL_FILES_ONLY

        if _LOCAL_FILES_ONLY.get():
            return True
    except Exception:  # noqa: BLE001
        pass
    if (os.environ.get("UNSLOTH_DIFFUSION_HEADER_FETCH") or "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    ):
        return True
    return (os.environ.get("HF_HUB_OFFLINE") or "").strip().lower() in ("1", "on", "true", "yes")


def _hub_header(
    repo_id: str, filename: str, revision: str, token: Any, size: int, deadline: float
) -> Optional[dict]:
    try:
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers

        from .prequant_header_probe import (
            MAX_SAFETENSORS_HEADER_BYTES,
            _hub_range_reader,
            _json_within,
        )

        reader = _hub_range_reader(
            hf_hub_url(repo_id, filename, revision=revision),
            dict(build_hf_headers(token=token)),
            deadline,
        )
        (length,) = struct.unpack("<Q", reader(0, 7))
        if not 0 < length <= MAX_SAFETENSORS_HEADER_BYTES or 8 + length > size:
            return None
        raw = reader(8, 8 + length - 1)
        if not _json_within(raw):
            return None
        header = json.loads(raw.decode("utf-8"))
        return header if isinstance(header, dict) else None
    except Exception:  # noqa: BLE001
        return None


MAX_INDEX_BYTES = 64 * 1024 * 1024


def _fetch_capped_text(
    session: Any,
    url: str,
    headers: dict,
    timeout: float,
    deadline: float,
    cap: int = MAX_INDEX_BYTES,
) -> str:
    """GET ``url`` as text, streamed: refused on a declared Content-Length over ``cap`` and stopped as soon as the
    body passes it, so an oversized (or lying) index is never buffered whole."""
    import time

    response = session.get(url, headers=headers, timeout=timeout, stream=True)
    try:
        response.raise_for_status()
        declared = (
            response.headers.get("Content-Length") if getattr(response, "headers", None) else None
        )
        if declared is not None and int(declared) > cap:
            raise ValueError("index too large")
        body = bytearray()
        for chunk in response.iter_content(chunk_size=65536):
            body.extend(chunk)
            if len(body) > cap:
                raise ValueError("index too large")
            if time.monotonic() > deadline:
                raise TimeoutError("index read deadline")
        return body.decode("utf-8")
    finally:
        try:
            response.close()
        except Exception:  # noqa: BLE001
            pass


def hub_component_sizes(
    repo_id: str,
    *,
    components: Optional[Iterable[str]] = None,
    revision: Optional[str] = None,
    token: Any = None,
    timeout_seconds: float = HUB_TIMEOUT_SECONDS,
) -> Optional[ComponentSizes]:
    """Component sizes of a Hub diffusers repo from ranged header reads (a few MB at most, no weights), read with
    ``token`` (the load's own token for a gated / private repo). Cached per resolved commit; an answer for a mutable ref
    is revalidated after ``HUB_REFRESH_SECONDS``, a failure retried after ``HUB_RETRY_SECONDS``. None offline, on any
    refusal, or past ``timeout_seconds``. Never raises."""
    if hub_disabled():
        return None
    import time

    wanted = set(components) if components is not None else None
    key = (
        repo_id,
        revision,
        tuple(sorted(wanted)) if wanted is not None else None,
        _token_id(token),
    )
    immutable = bool(revision and _SHA_RE.match(str(revision)))
    with _HUB_LOCK:
        if key in _HUB_CACHE:
            hit, stamp = _HUB_CACHE[key]
            age = time.monotonic() - stamp
            if hit is not None and (immutable or age < HUB_REFRESH_SECONDS):
                return hit
            if hit is None and age < HUB_RETRY_SECONDS:
                return None  # never a stall per load
    result = _hub_component_sizes(repo_id, wanted, revision, token, float(timeout_seconds))
    with _HUB_LOCK:
        _HUB_CACHE[key] = (result, time.monotonic())
    return result


def _hub_component_sizes(
    repo_id: str, wanted: Optional[set], revision: Optional[str], token: Any, timeout_seconds: float
) -> Optional[ComponentSizes]:
    import time

    deadline = time.monotonic() + timeout_seconds
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).model_info(
            repo_id, revision=revision, files_metadata=True, timeout=timeout_seconds
        )
    except Exception:  # noqa: BLE001 - offline, gated, missing: the table answers
        return None
    commit = getattr(info, "sha", None) or revision
    sha_key = (
        repo_id,
        commit,
        tuple(sorted(wanted)) if wanted is not None else None,
        _token_id(token),
    )
    with _HUB_LOCK:
        if commit and sha_key in _HUB_SHA_CACHE:
            return _HUB_SHA_CACHE[sha_key]  # the ref still points at a commit already read
    result = _read_hub_sizes(repo_id, info, commit, wanted, token, deadline, timeout_seconds)
    if result is not None and commit:
        with _HUB_LOCK:
            _HUB_SHA_CACHE[sha_key] = result
    return result


def _read_hub_sizes(
    repo_id: str,
    info: Any,
    commit: Optional[str],
    wanted: Optional[set],
    token: Any,
    deadline: float,
    timeout_seconds: float,
) -> Optional[ComponentSizes]:
    import time

    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session

    by_folder: dict[str, dict[str, int]] = {}
    for sib in getattr(info, "siblings", None) or ():
        parts = str(getattr(sib, "rfilename", "")).split("/")
        if len(parts) != 2:
            continue
        if wanted is not None and parts[0] not in wanted:
            continue
        by_folder.setdefault(parts[0], {})[parts[1]] = int(getattr(sib, "size", 0) or 0)
    out: dict[str, ComponentSize] = {}
    try:
        for folder, files in sorted(by_folder.items()):

            def _read_index(n: str, folder: str = folder) -> str:
                # fetched, never cached: a stray index in the HF cache would read as a started download
                return _fetch_capped_text(
                    get_session(),
                    hf_hub_url(repo_id, f"{folder}/{n}", revision=commit),
                    dict(build_hf_headers(token=token)),
                    max(1.0, deadline - time.monotonic()),
                    deadline,
                )

            if time.monotonic() > deadline:
                return None
            picked = _pick_files(list(files), _read_index, folder)
            if not picked:
                if any(n.endswith(".safetensors") or _INDEX_RE.match(n) for n in files):
                    return None  # safetensors weights this read cannot select (index unreadable / incomplete)
                continue
            size = _accumulate(
                folder,
                (
                    (
                        f,
                        _hub_header(
                            repo_id, f"{folder}/{f}", commit, token, files.get(f, 0), deadline
                        ),
                    )
                    for f in picked
                ),
            )
            if size is None:
                # A header that did not read makes the whole answer unknown, optional folders included: a partial
                # result would be cached as a smaller total.
                return None
            out[folder] = size
    except Exception:  # noqa: BLE001
        return None
    return ComponentSizes(out, source="hub", revision=commit) if out else None


def component_sizes(
    repo_or_dir: Any,
    *,
    components: Optional[Iterable[str]] = None,
    allow_hub: bool = False,
    cache_dir: Any = None,
    required: Optional[Iterable[str]] = None,
    token: Any = None,
) -> Optional[ComponentSizes]:
    """Local headers first (a path, or the repo's cached snapshot), then, with ``allow_hub``, ranged Hub reads (with
    ``token``). ``components`` are the folders counted when present; the local answer is complete once every
    ``required`` folder (default: all of ``components``) is there, so optional folders never send a cached model to
    the Hub."""
    if repo_or_dir is None:
        return None
    path = Path(str(repo_or_dir)).expanduser()
    if path.is_dir():
        return local_component_sizes(path, components)
    snap = local_snapshot_dir(str(repo_or_dir), cache_dir)
    local = local_component_sizes(snap, components) if snap is not None else None
    need = (
        tuple(required)
        if required is not None
        else (tuple(components) if components is not None else ())
    )
    if local is not None and all(c in local.components for c in need):
        return local
    if allow_hub:
        hub = hub_component_sizes(str(repo_or_dir), components=components, token=token)
        if hub is not None:
            return hub
    return local


# ---- video families: (denoiser(s), text encoders, VAE + audio) in bf16-resident decimal GB ----

# Component folder -> role of the family size triple. A folder not named here is outside the table (it is either
# not loaded by the pipeline or priced by its own term, e.g. LTX-2.5's latent upsampler).
VIDEO_ROLE_DENOISER = ("transformer", "transformer_2")
VIDEO_ROLE_TEXT = ("text_encoder", "text_encoder_2")
VIDEO_ROLE_VAE = ("vae", "audio_vae", "vocoder", "connectors")
# Per-family additions to the companion (third) term, as the table records them.
VIDEO_EXTRA_VAE_ROLE: dict[str, tuple[str, ...]] = {"ltx-2.5": ("latent_upsampler",)}
# Absolute GB tolerance under which a header-derived term and a table term agree (the table holds one decimal).
TABLE_ROUNDING_GB = 0.06

_MISMATCHES: dict = {}


def table_mismatches() -> dict:
    """Header-vs-table disagreements seen this process: ``{(family, repo): (header_triple, table_triple)}``."""
    return dict(_MISMATCHES)


def video_required_roles(fam: Any) -> tuple[str, ...]:
    """The folders without which ``video_header_components_gb`` cannot answer: the denoiser (both experts for MoE),
    the text encoder, the VAE and, for an audio family, the audio VAE. Connectors, vocoder, upsampler, a second text
    encoder are optional: counted when present, never a reason to go to the Hub."""
    denoiser = getattr(fam, "denoiser_attr", None) or "transformer"
    roles = [denoiser]
    if bool(getattr(fam, "is_moe", False)):
        roles.append("transformer_2")
    roles += ["text_encoder", "vae"]
    if bool(getattr(fam, "has_audio", False)):
        roles.append("audio_vae")
    return tuple(dict.fromkeys(roles))


def video_header_terms(
    fam: Any, sizes: Optional[ComponentSizes]
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Each term of the family triple from header sizes, or None for a term whose required folders are not all
    known (e.g. a ``.bin`` text encoder beside a safetensors transformer). The denoiser term needs every expert
    (``transformer_2`` counts only for an MoE family), the text term ``text_encoder``, the VAE term ``vae`` (+
    ``audio_vae`` for an audio family); optional folders add when present. Float terms are cast to bf16, a
    ``vae_force_fp32`` family keeps its VAE at fp32 (as the table records it)."""
    if sizes is None:
        return (None, None, None)
    name = str(getattr(fam, "name", "") or "")
    moe = bool(getattr(fam, "is_moe", False))
    primary = getattr(fam, "denoiser_attr", None) or "transformer"
    denoiser_names = tuple(dict.fromkeys((primary, "transformer_2") if moe else (primary,)))
    denoiser = (
        sizes.resident_gb(denoiser_names)
        if all(sizes.get(n) is not None for n in denoiser_names)
        else None
    )
    text = (
        sizes.resident_gb([n for n in VIDEO_ROLE_TEXT if sizes.get(n) is not None])
        if sizes.get("text_encoder") is not None
        else None
    )
    vae_names = VIDEO_ROLE_VAE + VIDEO_EXTRA_VAE_ROLE.get(name, ())
    vae_ok = sizes.get("vae") is not None and (
        not bool(getattr(fam, "has_audio", False)) or sizes.get("audio_vae") is not None
    )
    vae = None
    if vae_ok:
        vae_itemsize = 4 if bool(getattr(fam, "vae_force_fp32", False)) else 2
        vae = (
            sum(
                sizes.components[n].resident_bytes(vae_itemsize if n == "vae" else 2)
                for n in vae_names
                if sizes.get(n) is not None
            )
            / _GB
        )
    return (denoiser, text, vae)


def video_header_components_gb(
    fam: Any, sizes: Optional[ComponentSizes]
) -> Optional[tuple[float, float, float]]:
    """The family triple from header sizes, or None unless every term is known (``video_header_terms``)."""
    terms = video_header_terms(fam, sizes)
    return None if any(t is None for t in terms) else terms  # type: ignore[return-value]


IMAGE_ROLE_DENOISER = ("transformer", "transformer_2", "unconditional_transformer", "unet", "prior")


def image_header_components_gb(
    sizes: Optional[ComponentSizes],
) -> Optional[tuple[float, float, float]]:
    """``(denoiser(s), text encoders, vae)`` bf16-resident GB of an image pipeline, the ``_FAMILY_BF16_GB`` layout;
    None unless a denoiser, a text encoder and a VAE are all present."""
    if sizes is None:
        return None
    denoisers = [n for n in IMAGE_ROLE_DENOISER if sizes.get(n) is not None]
    text = [n for n in sizes.names() if re.match(r"^text_encoder(?:_\d+)?$", n)]
    if not denoisers or not text or sizes.get("vae") is None:
        return None
    return (sizes.resident_gb(denoisers), sizes.resident_gb(text), sizes.resident_gb(("vae",)))


def reconcile_with_table(
    header: Optional[tuple[float, float, float]],
    table: Optional[tuple[float, float, float]],
    *,
    is_base: bool,
    key: Any = None,
) -> Optional[tuple[float, float, float]]:
    """Pick the triple the planners see. No header: the table. No table: the header (rounded to one decimal, the
    table's precision). Both: the table where they agree within rounding; on a disagreement the table still wins for
    the family's own base repo (the disagreement is recorded for review, never applied silently). A different repo (a
    variant / finetune / local copy the table does not describe) takes the larger of the two per term: a variant
    bigger than its base is priced at its own size, a smaller or same-size one keeps today's table plan.

    A header term may be None (unknown, e.g. a ``.bin`` component): that term takes the table, the known ones are
    reconciled on their own, so a mixed-format variant keeps its known larger denoiser. Without a table every term
    must be known."""
    if header is None or all(h is None for h in header):
        return table
    if table is None:
        if any(h is None for h in header):
            return None
        return tuple(round(float(v), 1) for v in header)  # type: ignore[return-value]
    header = tuple(t if h is None else h for h, t in zip(header, table))
    rounded = tuple(round(float(v), 1) for v in header)
    agree = all(abs(float(h) - float(t)) <= TABLE_ROUNDING_GB for h, t in zip(header, table))
    if agree:
        return table
    if key is not None and key not in _MISMATCHES:
        _MISMATCHES[key] = (rounded, tuple(table))
        logger.info(
            "diffusion.sizes: %s headers %s disagree with the table %s; using the %s",
            key,
            rounded,
            tuple(table),
            "table" if is_base else "larger per term",
        )
    if is_base:
        return tuple(table)  # type: ignore[return-value]
    return tuple(max(float(h), float(t)) for h, t in zip(rounded, table))  # type: ignore[return-value]


def reconcile_scalar(
    header: Optional[float], table: float, *, is_base: bool, key: Any = None
) -> float:
    """``reconcile_with_table`` for one term (a per-component constant such as LTX-2.5's latent upsampler)."""
    out = reconcile_with_table(
        None if header is None else (float(header),), (float(table),), is_base=is_base, key=key
    )
    return float(out[0]) if out else float(table)


def video_components_gb(
    fam: Any,
    repo: Any = None,
    *,
    allow_hub: bool = False,
    token: Any = None,
) -> Optional[tuple[float, float, float]]:
    """``(denoiser(s), text encoders, VAE + audio)`` bf16-resident decimal GB for ``fam`` loaded from ``repo``
    (default: the family's base repo), from safetensors headers (local snapshot, then ranged Hub reads with the load's
    ``token`` when ``allow_hub``) reconciled with ``fam.bf16_components_gb``. Never raises; None when neither
    answers."""
    table = getattr(fam, "bf16_components_gb", None)
    try:
        base = getattr(fam, "base_repo", None)
        target = repo or base
        if not target or getattr(fam, "modular_workflow", None):
            return table  # modular (H3) layouts are not a diffusers component tree
        is_base = str(target).lower() == str(base or "").lower()
        wanted = (
            *(VIDEO_ROLE_DENOISER if getattr(fam, "is_moe", False) else VIDEO_ROLE_DENOISER[:1]),
            *(() if getattr(fam, "denoiser_attr", "transformer") in VIDEO_ROLE_DENOISER else (fam.denoiser_attr,)),
            *VIDEO_ROLE_TEXT,
            *VIDEO_ROLE_VAE,
            *VIDEO_EXTRA_VAE_ROLE.get(str(getattr(fam, "name", "")), ()),
        )
        sizes = component_sizes(
            target,
            components=wanted,
            required=video_required_roles(fam),
            allow_hub=allow_hub and (table is None or not is_base),
            token=token,
        )
        header = video_header_terms(fam, sizes)
        return reconcile_with_table(
            header, table, is_base=is_base, key=(getattr(fam, "name", None), str(target))
        )
    except Exception:  # noqa: BLE001 - sizing aid: the table stands
        return table


def component_gb(
    repo: Any, component: str, *, itemsize: int = 2, default: Optional[float] = None
) -> Optional[float]:
    """One component's resident GB from local headers (no network), else ``default``."""
    try:
        sizes = component_sizes(repo, components=(component,))
        size = sizes.get(component) if sizes is not None else None
        return size.resident_gb(itemsize) if size is not None else default
    except Exception:  # noqa: BLE001
        return default
