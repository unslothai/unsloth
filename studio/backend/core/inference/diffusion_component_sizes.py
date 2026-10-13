# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Component sizes of a diffusers pipeline from safetensors headers; the family tables stay the offline fallback."""

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

MAX_HEADER_BYTES = 256 * 1024 * 1024
# Bits per element of every floating dtype safetensors names. Float8 / float6 / float4 / MX scales count as float:
# loaded at bf16 they occupy 2 bytes per parameter. Sub-byte types are packed: the header ``shape`` is the logical
# element count and the byte span is numel * bits / 8 (an F4 [4, 16] tensor spans 32 bytes), so parameters are
# counted as bytes * 8 / bits.
FLOAT_BITS = {
    "F64": 64,
    "F32": 32,
    "F16": 16,
    "BF16": 16,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E8M0": 8,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "F4": 4,
}
FLOAT_WIDTH = {k: v / 8 for k, v in FLOAT_BITS.items()}
_GB = 1000.0**3

ReadRange = Callable[[int, int], bytes]


def parse_safetensors_header(
    read: Callable[[int], bytes], *, max_bytes: int = MAX_HEADER_BYTES
) -> dict:
    """Raises ValueError on a non-safetensors file or implausible header length."""
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
    """None when unreadable. Never raises."""
    try:
        with open(path, "rb") as fh:
            return parse_safetensors_header(fh.read, max_bytes=max_bytes)
    except Exception:  # noqa: BLE001 - unreadable / crafted header: no claim about this file
        return None


def tensor_entries(header: Optional[dict]) -> Iterable[tuple[str, dict]]:
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

    name: str
    stored_bytes: int
    float_params: int
    nonfloat_bytes: int
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

    components: dict
    source: str = "local"
    revision: Optional[str] = None
    # Component names model_index.json declares (None when it was not read).
    manifest: Optional[frozenset] = None
    # Folders holding weights whose size could not be read (a .bin checkpoint, an unreadable header or index).
    unknown: frozenset = frozenset()

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
            if dtype in FLOAT_BITS:
                bits = FLOAT_BITS[dtype]
                float_params += numel if bits >= 8 else (nbytes * 8) // bits
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


# The loader's default weight stems (diffusion.py _DEFAULT_PIPELINE_WEIGHT_RE), safetensors only.
_DEFAULT_WEIGHT_RE = re.compile(
    r"^(?:diffusion_pytorch_model|pytorch_model|model)(?:-\d{5}-of-\d{5})?\.safetensors$"
)
_SHARD_RE = re.compile(r"^.+-\d{5}-of-\d{5}\.safetensors$")
_INDEX_RE = re.compile(
    r"^(?:diffusion_pytorch_model|pytorch_model|model)\.safetensors\.index\.json$"
)


_TRANSFORMERS_FOLDER_RE = re.compile(r"^(?:text_encoder|image_encoder|prompt_enhancer)(?:_\d+)?$")


def _stem_order(folder: str) -> tuple[str, ...]:
    """LTX-2's text_encoder/ ships both stems; only the preferred one loads. ``pytorch_model`` is the legacy stem."""
    if _TRANSFORMERS_FOLDER_RE.match(folder or ""):
        return ("model", "diffusion_pytorch_model", "pytorch_model")
    return ("diffusion_pytorch_model", "model", "pytorch_model")


def select_weight_files(
    names: Iterable[str], index_files: Optional[Iterable[str]] = None, folder: str = ""
) -> list[str]:
    """Listed ``index_files`` win over stray shards of an older layout."""
    names = [n for n in names if n.endswith(".safetensors")]
    if index_files:
        listed = sorted(set(index_files))
        if all(n in names for n in listed):
            return listed
        return []  # an index naming a file not present: incomplete, unknown
    # Shards without an index are unknown, never the sum of whatever is present.
    for stem in _stem_order(folder):
        if f"{stem}.safetensors" in names:
            return [f"{stem}.safetensors"]
        if any(_SHARD_RE.match(n) and n.startswith(stem + "-") for n in names):
            return []
    # Nothing else is selected: from_pretrained reads only these stems (Studio passes no variant, so a
    # diffusion_pytorch_model.fp16.safetensors twin is not read either), never e.g. adapter_model.safetensors.
    return []


_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".safetensors.index.json", ".bin.index.json")


def _holds_weights(names: Iterable[str]) -> bool:
    return any(str(n).endswith(_WEIGHT_SUFFIXES) for n in names)


def _read_capped_file(path: Path, cap: int) -> str:
    if path.stat().st_size > cap:
        raise ValueError(f"{path.name} is larger than {cap} bytes")
    return path.read_text(encoding="utf-8")


def _index_shards(text: str) -> Optional[list[str]]:
    try:
        weight_map = json.loads(text).get("weight_map") or {}
        return sorted(set(str(v) for v in weight_map.values()))
    except Exception:  # noqa: BLE001
        return None


def _pick_files(
    names: Iterable[str], read_text: Callable[[str], str], folder: str = ""
) -> list[str]:
    """Only the loader's PREFERRED index counts (it never falls back to a stale second one); unreadable or
    naming a missing shard makes the component unknown, never a smaller subset."""
    names = list(names)
    # Same-stem order follows the loader: transformers (5.5 modeling_utils._get_resolved_checkpoint_files) takes an
    # unsharded model.safetensors before the index; diffusers (ModelMixin.from_pretrained, _fetch_index_file) takes
    # the index first.
    unsharded_first = bool(_TRANSFORMERS_FOLDER_RE.match(folder or ""))
    for stem in _stem_order(folder):
        index = f"{stem}.safetensors.index.json"
        if unsharded_first and f"{stem}.safetensors" in names:
            return [f"{stem}.safetensors"]
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
            return select_weight_files(
                [n for n in names if n.startswith((stem + ".", stem + "-"))], folder=folder
            )
    return select_weight_files(names, folder=folder)


def local_component_sizes(
    root: Any, components: Optional[Iterable[str]] = None
) -> Optional[ComponentSizes]:
    """None when unreadable or weightless. Never raises."""
    try:
        base = Path(root)
        if not base.is_dir():
            return None
        wanted = set(components) if components is not None else None
        out: dict[str, ComponentSize] = {}
        unknown: set = set()
        for sub in sorted(p for p in base.iterdir() if p.is_dir()):
            if wanted is not None and sub.name not in wanted:
                continue
            names = [p.name for p in sub.iterdir() if p.is_file()]
            files = _pick_files(
                names, lambda n, sub=sub: _read_capped_file(sub / n, MAX_INDEX_BYTES), sub.name
            )
            size = (
                _accumulate(sub.name, ((f, read_safetensors_header(sub / f)) for f in files))
                if files
                else None
            )
            if size is not None:
                out[sub.name] = size
            elif _holds_weights(names):
                unknown.add(sub.name)
        if not out:
            return None
        return ComponentSizes(
            out, source="local", manifest=_manifest_names(base), unknown=frozenset(unknown)
        )
    except Exception:  # noqa: BLE001
        return None


def _manifest_from_text(text: str) -> Optional[frozenset]:
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return None
        return frozenset(
            k
            for k, v in data.items()
            if not k.startswith("_") and isinstance(v, (list, tuple)) and v and v[0] is not None
        )
    except Exception:  # noqa: BLE001
        return None


def _manifest_names(root: Path) -> Optional[frozenset]:
    """Components ``model_index.json`` loads (an entry set to null is not loaded)."""
    try:
        path = root / "model_index.json"
        if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
            return None
        return _manifest_from_text(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def active_cache_dir() -> Optional[str]:
    """Studio's LIVE hub cache root, not huggingface_hub's import-time default."""
    try:
        from utils.hf_cache_settings import active_hf_hub_cache

        return active_hf_hub_cache()
    except Exception:  # noqa: BLE001
        return None


def local_snapshot_dir(repo_id: str, cache_dir: Any = None) -> Optional[Path]:
    try:
        from .diffusion_small_host import resolve_snapshot_dir

        return resolve_snapshot_dir(repo_id, cache_dir if cache_dir is not None else active_cache_dir())
    except Exception:  # noqa: BLE001
        return None



HUB_TIMEOUT_SECONDS = 20.0
HUB_RETRY_SECONDS = 600.0
# Mutable-ref answers revalidate after this; headers are re-read only if the commit changed.
HUB_REFRESH_SECONDS = 600.0
_HUB_CACHE: dict = {}
_HUB_SHA_CACHE: dict = {}
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _token_id(token: Any) -> Optional[str]:
    """Token digest: a gated repo read with one token is never served to a load without it."""
    if not token or not isinstance(token, str):
        return None if not token else "implicit"
    import hashlib

    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


_HUB_LOCK = threading.Lock()


def hub_disabled() -> bool:
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
MAX_MANIFEST_BYTES = 1 << 20


def _fetch_capped_text(
    session: Any,
    url: str,
    headers: dict,
    timeout: float,
    deadline: float,
    cap: int = MAX_INDEX_BYTES,
) -> str:
    """Refuses past ``cap`` while streaming, so an oversized (or lying) index is never buffered whole. Works on
    both clients huggingface_hub ships: httpx (1.x, ``Client.stream``, redirects followed to the CDN) and requests
    (0.x, ``stream=True``); see ``diffusion_compat._ranged_stream``."""
    import time

    from .diffusion_compat import _iter_body

    if callable(getattr(session, "stream", None)):
        opened = session.stream(
            "GET", url, headers=headers, timeout=timeout, follow_redirects=True
        )
    else:
        opened = session.get(url, headers=headers, timeout=timeout, stream=True)
    with opened as response:
        response.raise_for_status()
        declared = (
            response.headers.get("Content-Length") if getattr(response, "headers", None) else None
        )
        if declared is not None and int(declared) > cap:
            raise ValueError("index too large")
        body = bytearray()
        for chunk in _iter_body(response, 65536):
            body.extend(chunk)
            if len(body) > cap:
                raise ValueError("index too large")
            if time.monotonic() > deadline:
                raise TimeoutError("index read deadline")
        return body.decode("utf-8")


_REF_CACHE: dict = {}


def resolve_ref(
    repo_id: str,
    revision: Optional[str] = None,
    *,
    token: Any = None,
    timeout_seconds: float = HUB_TIMEOUT_SECONDS,
) -> Optional[str]:
    """The commit sha ``revision`` (default ``main``) points at now, from one metadata call cached for
    ``HUB_REFRESH_SECONDS`` (a failure for ``HUB_RETRY_SECONDS``). A pinned sha is returned as is. None when the Hub
    is off or does not answer. Never raises."""
    if revision and _SHA_RE.match(str(revision)):
        return str(revision)
    if hub_disabled():
        return None
    import time

    key = (repo_id, revision, _token_id(token))
    with _HUB_LOCK:
        if key in _REF_CACHE:
            sha, stamp = _REF_CACHE[key]
            age = time.monotonic() - stamp
            if (sha is not None and age < HUB_REFRESH_SECONDS) or (
                sha is None and age < HUB_RETRY_SECONDS
            ):
                return sha
    sha = None
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).model_info(repo_id, revision=revision, timeout=timeout_seconds)
        sha = getattr(info, "sha", None)
    except Exception:  # noqa: BLE001
        sha = None
    _note_ref(repo_id, revision, token, sha)
    return sha


def _note_ref(repo_id: str, revision: Optional[str], token: Any, sha: Optional[str]) -> None:
    import time

    with _HUB_LOCK:
        _REF_CACHE[(repo_id, revision, _token_id(token))] = (sha, time.monotonic())


def hub_component_sizes(
    repo_id: str,
    *,
    components: Optional[Iterable[str]] = None,
    revision: Optional[str] = None,
    token: Any = None,
    timeout_seconds: float = HUB_TIMEOUT_SECONDS,
) -> Optional[ComponentSizes]:
    """Ranged header reads with the load's ``token``, cached per resolved commit. Never raises."""
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
    if getattr(info, "sha", None):
        _note_ref(repo_id, revision, token, info.sha)  # the same resolve resolve_ref would make
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
    # The manifest first: a folder model_index.json omits or sets to null is not loaded, so its failure is not ours.
    manifest = None
    if any(str(getattr(s, "rfilename", "")) == "model_index.json" for s in info.siblings or ()):
        try:
            manifest = _manifest_from_text(
                _fetch_capped_text(
                    get_session(),
                    hf_hub_url(repo_id, "model_index.json", revision=commit),
                    dict(build_hf_headers(token=token)),
                    max(1.0, deadline - time.monotonic()),
                    deadline,
                    cap=MAX_MANIFEST_BYTES,
                )
            )
        except Exception:  # noqa: BLE001 - no manifest: every folder counts as loaded
            manifest = None
    out: dict[str, ComponentSize] = {}
    unknown: set = set()
    try:
        for folder, files in sorted(by_folder.items()):
            if manifest is not None and folder not in manifest and folder not in _LOADED_OUTSIDE_MANIFEST:
                continue

            def _read_index(n: str, folder: str = folder) -> str:
                # never from the HF cache: a stray index there would read as a started download
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
                if _holds_weights(files):
                    unknown.add(folder)  # a loaded .bin component: its term is unknown, never smaller
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
                # any failed header of a loaded folder voids the answer: a partial one would be cached smaller
                return None
            out[folder] = size
    except Exception:  # noqa: BLE001
        return None
    if not out:
        return None
    return ComponentSizes(
        out, source="hub", revision=commit, manifest=manifest, unknown=frozenset(unknown)
    )


def component_sizes(
    repo_or_dir: Any,
    *,
    components: Optional[Iterable[str]] = None,
    allow_hub: bool = False,
    cache_dir: Any = None,
    required: Optional[Iterable[str]] = None,
    token: Any = None,
) -> Optional[ComponentSizes]:
    """Local headers, then Hub; optional folders (not in ``required``) never send a cached model to the Hub."""
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
        if not allow_hub or snap is None or hub_disabled():
            return local
        # online: the cached snapshot answers only while the ref still points at it (TTL-cached resolve)
        # an unresolved ref (head None) is not a match: probe the Hub, and keep the local sizes only if that fails too
        head = resolve_ref(str(repo_or_dir), token=token)
        if head is not None and head == snap.name:
            return local
    if allow_hub:
        hub = hub_component_sizes(str(repo_or_dir), components=components, token=token)
        if hub is not None:
            return hub
    return local



# Folders not named here are outside the table (unloaded, or priced separately like LTX-2.5's upsampler).
VIDEO_ROLE_DENOISER = ("transformer", "transformer_2")
VIDEO_ROLE_TEXT = ("text_encoder", "text_encoder_2", "text_encoder_3")
VIDEO_ROLE_VAE = ("vae", "audio_vae", "vocoder", "connectors")
VIDEO_EXTRA_VAE_ROLE: dict[str, tuple[str, ...]] = {"ltx-2.5": ("latent_upsampler",)}
# Studio attaches these itself, so model_index.json never names them (LTX-2.5's latent_upsampler/).
_LOADED_OUTSIDE_MANIFEST = frozenset(f for roles in VIDEO_EXTRA_VAE_ROLE.values() for f in roles)
# Companion folders and when a family loads them; one that is present is counted only when its family rule says so
# or model_index.json names it (a leftover folder in a repo is not a loaded component).
VIDEO_COMPANION_RULES: dict[str, Callable[[Any], bool]] = {
    "audio_vae": lambda fam: bool(getattr(fam, "has_audio", False)),
    "vocoder": lambda fam: bool(getattr(fam, "has_audio", False)),
    "connectors": lambda fam: str(getattr(fam, "name", "") or "").startswith("ltx"),
    # HunyuanVideo-1.5 loads its ByT5 glyph encoder as text_encoder_2; any other extra encoder needs the manifest.
    "text_encoder_2": lambda fam: str(getattr(fam, "name", "") or "").startswith("hunyuanvideo"),
    "text_encoder_3": lambda fam: False,
}


def _companion_loaded(fam: Any, sizes: ComponentSizes, name: str) -> bool:
    rule = VIDEO_COMPANION_RULES.get(name)
    if rule is None:
        return True
    if rule(fam):
        return True
    return sizes.manifest is not None and name in sizes.manifest


TABLE_ROUNDING_GB = 0.06

_MISMATCHES: dict = {}


def table_mismatches() -> dict:
    return dict(_MISMATCHES)


def video_required_roles(fam: Any) -> tuple[str, ...]:
    """Connectors, vocoder, upsampler, a second text encoder are optional: never a reason to go to the Hub."""
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
    """None for a term with a missing required folder (e.g. a ``.bin`` text encoder)."""
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
    def _loaded_unknown(names: Iterable[str]) -> bool:
        # a folder the family / manifest loads whose size did not read: the term is unknown, never the rest summed
        return any(n in sizes.unknown and _companion_loaded(fam, sizes, n) for n in names)

    text = (
        sizes.resident_gb(
            [
                n
                for n in VIDEO_ROLE_TEXT
                if sizes.get(n) is not None and _companion_loaded(fam, sizes, n)
            ]
        )
        if sizes.get("text_encoder") is not None and not _loaded_unknown(VIDEO_ROLE_TEXT)
        else None
    )
    vae_names = VIDEO_ROLE_VAE + VIDEO_EXTRA_VAE_ROLE.get(name, ())
    vae_ok = (
        sizes.get("vae") is not None
        and (not bool(getattr(fam, "has_audio", False)) or sizes.get("audio_vae") is not None)
        and not _loaded_unknown(vae_names)
    )
    vae = None
    if vae_ok:
        vae_itemsize = 4 if bool(getattr(fam, "vae_force_fp32", False)) else 2
        vae = (
            sum(
                sizes.components[n].resident_bytes(vae_itemsize if n == "vae" else 2)
                for n in vae_names
                if sizes.get(n) is not None and _companion_loaded(fam, sizes, n)
            )
            / _GB
        )
    return (denoiser, text, vae)


def video_header_components_gb(
    fam: Any, sizes: Optional[ComponentSizes]
) -> Optional[tuple[float, float, float]]:
    terms = video_header_terms(fam, sizes)
    return None if any(t is None for t in terms) else terms  # type: ignore[return-value]


IMAGE_ROLE_DENOISER = ("transformer", "transformer_2", "unconditional_transformer", "unet", "prior")


def image_header_components_gb(
    sizes: Optional[ComponentSizes],
) -> Optional[tuple[float, float, float]]:
    """``_FAMILY_BF16_GB`` layout; None unless denoiser, text encoder and VAE are all present."""
    if sizes is None:
        return None
    denoisers = [n for n in IMAGE_ROLE_DENOISER if sizes.get(n) is not None]
    text = [
        n
        for n in sizes.names()
        if re.match(r"^text_encoder(?:_\d+)?$", n)
        and (n == "text_encoder" or sizes.manifest is None or n in sizes.manifest)
    ]
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
    """Base repo: the table wins (disagreements recorded, never applied). Other repo: per-term max of the two.
    An unknown header term takes the table; without a table every term must be known."""
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


def video_components_gb(
    fam: Any,
    repo: Any = None,
    *,
    allow_hub: bool = False,
    token: Any = None,
    details: Optional[dict] = None,
) -> Optional[tuple[float, float, float]]:
    """Header sizes reconciled with ``fam.bf16_components_gb``. Never raises; None when neither answers.
    ``details`` (filled when given): ``sizes`` (the snapshot the answer came from) and ``sources``, per term
    "header" or "table", so a caller pricing a part of a term reads it from the same answer."""
    table = getattr(fam, "bf16_components_gb", None)
    if details is not None:
        details.update(sizes=None, sources=("table",) * 3)
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
        result = reconcile_with_table(
            header, table, is_base=is_base, key=(getattr(fam, "name", None), str(target))
        )
        if details is not None and result is not None:
            sources = tuple(
                "table" if table is not None and float(r) == float(t) else "header"
                for r, t in zip(result, table or (None,) * 3)
            )
            details.update(sizes=sizes, sources=sources)
        return result
    except Exception:  # noqa: BLE001 - sizing aid: the table stands
        return table


def term_part_gb(details: Optional[dict], term: int, folder: str, table_gb: float) -> float:
    """The share of term ``term`` that ``folder`` holds, from the SAME answer the term came from: its header size
    when the term is header-derived, else the table's constant for it."""
    try:
        if details and details.get("sources", ())[term] == "header":
            size = details["sizes"].get(folder) if details.get("sizes") is not None else None
            if size is not None:
                return size.resident_gb(2)
    except Exception:  # noqa: BLE001
        pass
    return float(table_gb)

