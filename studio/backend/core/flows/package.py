# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Share and import flows.

Export modes: ``json`` (the flow alone), ``package`` (a zip with the media its inputs use) and ``baked`` (also every
pre-generated output, so it plays with no models). Secrets, provider ids and local Library or gallery ids never leave:
media is embedded by content hash instead.

Import never runs or downloads anything. A zip may hold only ``flow.json``, ``preview.png``, ``README.md`` and
``assets/<sha256>.<ext>``; anything else, a path that climbs out, a link, a duplicate name, an oversized or
suspiciously compressible entry, or an asset whose bytes do not match its name or type is refused."""

from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import zipfile
from typing import Any, Awaitable, Callable
from urllib.parse import unquote_plus

from core.flows import assets as asset_store
from core.flows import glb, schema
from core.flows.compiler import parse_document
from core.flows import replication
from core.flows.recipe_share import missing_seed, share_flow
from core.flows.specs import SPECS_BY_TYPE

MAX_ENTRIES = schema.MAX_ASSETS + 8
MAX_RATIO = 200
FIXED_ENTRIES = ("flow.json", "preview.png", "README.md")
_ASSET_NAME = re.compile(r"^assets/([0-9a-f]{64})\.([a-z0-9]{2,5})$")
# A credential term ending the key, also after a prefix: openai_api_key, aws_secret_access_key, x-api-key.
_SECRET_KEY = re.compile(
    r"(^|[_-])(api[_-]?key(_env)?|secret([_-]?(access[_-]?)?key)?|access[_-]?key|private[_-]?key|password|passwd"
    r"|credentials?|(hf|access|auth|bearer|refresh|github|api|session)[_-]?token|token|client[_-]?secret"
    r"|authorization|provider[_-]?id|extra[_-]?headers)$",
    re.I,
)
_SECRET_VALUE = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,})"
)
_PLAIN_HEADERS = ("accept", "accept-language", "content-type", "user-agent")
_LOCAL_REFS = ("image:", "audio:", "video:", "library:", "voice:")

ReadMedia = Callable[[dict], Awaitable[tuple[bytes, str]]]


class PackageError(Exception):
    pass


def _local_url(url: Any) -> bool:
    return isinstance(url, str) and url.startswith("/api/") and "//" not in url and "\\" not in url


def scrub(value: Any) -> Any:
    """Drop secret-looking keys, blank secret-looking strings and media URLs that leave this Studio, recursively."""
    if isinstance(value, dict):
        found = {
            k: scrub(v)
            for k, v in value.items()
            if not (isinstance(k, str) and _SECRET_KEY.search(k))
        }
        # Request headers carry credentials under any name (Cookie, X-Custom-Auth): keep the names, drop the values.
        if isinstance(found.get("headers"), dict):
            found["headers"] = {
                k: v if str(k).lower() in _PLAIN_HEADERS else ""
                for k, v in found["headers"].items()
            }
        # The player fetches media URLs with the user's token; a shared flow must not point it at another origin.
        if _is_media(found) and "url" in found and not _local_url(found["url"]):
            found.pop("url")
        return found
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        if _SECRET_VALUE.search(value):
            value = _SECRET_VALUE.sub("", value)
        if "://" in value and ("?" in value or "@" in value):
            value = _SECRET_QUERY.sub(_blank_query, _URL_USERINFO.sub(r"\1", value))
    return value


_URL_USERINFO = re.compile(r"(https?://)[^/@\s?#]+@", re.I)
_SECRET_QUERY = re.compile(r"([?&])([^=&#\s]+)=([^&#\s]*)")
_SIGNED_QUERY = re.compile(r"(^|[_-]|%2d|%5f)(signature|credential|security[_-]?token)$", re.I)


def _blank_query(match: "re.Match") -> str:
    name = match.group(2)
    # Servers decode the name (api%5Fkey is api_key); signed URLs carry live credentials (X-Amz-Signature, ...).
    plain = unquote_plus(name)
    if (
        _SECRET_KEY.search(plain)
        or _SIGNED_QUERY.search(plain)
        or plain.lower() in ("key", "sig", "signature", "code")
    ):
        return f"{match.group(1)}{name}="
    return match.group(0)


def _is_media(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("kind") in ("image", "audio", "video", "document", "model")
        and isinstance(value.get("ref"), str)
    )


def _is_data_asset(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("kind") in ("table", "heatmap", "tensor")
        and isinstance(value.get("ref"), str)
        and asset_store.parse_ref(value["ref"]) is not None
    )


async def _embed(value: Any, read_media: ReadMedia, files: dict[str, tuple[bytes, str]]) -> Any:
    if isinstance(value, list):
        return [await _embed(v, read_media, files) for v in value]
    if not isinstance(value, dict):
        return value
    if _is_media(value):
        digest = asset_store.parse_ref(value["ref"])
        if digest is not None:
            got = asset_store.read(digest)
            if got is None:
                return value
            data, mime = got
        else:
            data, mime = await read_media(value)
            mime = asset_store.sniff_mime(data) or mime
            digest = hashlib.sha256(data).hexdigest()
        files[digest] = (data, mime)
        embedded = {k: v for k, v in value.items() if k not in ("url",)}
        embedded["ref"] = f"sha256:{digest}"
        embedded["url"] = asset_store.url_for(digest)
        embedded["mime"] = mime
        return embedded
    if _is_data_asset(value):
        digest = asset_store.parse_ref(value["ref"])
        got = asset_store.read(digest)
        if got is None:
            return value
        data, mime = got
        try:
            parsed = json.loads(data)
        except (UnicodeDecodeError, ValueError):
            parsed = None
        clean = scrub(parsed)
        if parsed is not None and clean != parsed:
            data = json.dumps(clean, separators = (",", ":")).encode()
            digest = hashlib.sha256(data).hexdigest()
            value = {**value, "ref": f"sha256:{digest}"}
            if "url" in value:
                value["url"] = asset_store.url_for(digest)
        files[digest] = (data, mime)
        return value
    return {k: await _embed(v, read_media, files) for k, v in value.items()}


_LOCAL_SEEDS = ("local", "unstructured")
_UPLOAD_UI = (
    "unstructured_file_ids",
    "unstructured_file_names",
    "unstructured_file_sizes",
    "unstructured_upload_uid",
    "local_file_name",
)


def _drop_local_seeds(doc: dict) -> None:
    """A recipe group's seed preview rows, and an uploaded seed's paths and upload ids, stay here."""
    for group in doc.get("groups") or []:
        if not isinstance(group, dict) or group.get("kind") != "recipe":
            continue
        if isinstance(group.get("ui"), dict):
            group["ui"].pop("seed_preview_rows", None)
        seed = (group.get("config") or {}).get("seed_config")
        source = seed.get("source") if isinstance(seed, dict) else None
        if not isinstance(source, dict) or source.get("seed_type") not in _LOCAL_SEEDS:
            continue
        source.pop("path", None)
        source.pop("paths", None)
        if isinstance(group.get("ui"), dict):
            for key in _UPLOAD_UI:
                group["ui"].pop(key, None)


def _drop_local(value: Any) -> Any:
    if isinstance(value, list):
        return [
            _drop_local(v) for v in value if not (_is_media(v) and v["ref"].startswith(_LOCAL_REFS))
        ]
    if isinstance(value, dict):
        if _is_media(value) and value["ref"].startswith(_LOCAL_REFS):
            return None
        return {k: _drop_local(v) for k, v in value.items()}
    return value


DEFAULT_LICENSE = "CC-BY-4.0"
MAX_LINEAGE = 20


def record_lineage(doc: dict) -> None:
    """Before an imported flow gets its new id: remember the flow it came from (Scratch drops credit after one hop)."""
    meta = doc.setdefault("meta", {})
    parent = {"id": doc.get("id"), "name": doc.get("name")}
    if isinstance(meta.get("author"), str):
        parent["author"] = meta["author"][:200]
    chain = [p for p in meta.get("derived_from") or [] if isinstance(p, dict)]
    meta["derived_from"] = [parent, *chain][:MAX_LINEAGE]


def _uses_models(doc: dict) -> bool:
    return any(
        (SPECS_BY_TYPE.get(n.get("type")) or {}).get("lane") == "gpu" for n in doc.get("nodes", [])
    )


async def export_flow(doc: dict, mode: str, read_media: ReadMedia) -> tuple[bytes, str, str]:
    """(bytes, filename, media type) for ``mode`` json, package or baked."""
    doc = scrub(share_flow(json.loads(json.dumps(doc))))
    _drop_local_seeds(doc)
    meta = dict(doc.get("meta") or {})
    meta["created_with"] = "Unsloth Studio"
    meta.setdefault("license", DEFAULT_LICENSE)
    meta.setdefault("ai_generated", _uses_models(doc))
    doc["meta"] = meta
    replication.stamp(doc)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", doc.get("name") or "flow").strip("-").lower() or "flow"
    if mode == "json":
        doc["baked"] = {}
        doc["assets"] = []
        for node in doc.get("nodes", []):
            node["params"] = _drop_local(node.get("params") or {})
        if doc.get("params"):
            doc["params"] = _drop_local(doc["params"])
        return json.dumps(doc, indent = 1).encode(), f"{slug}.json", "application/json"
    files: dict[str, tuple[bytes, str]] = {}
    for node in doc.get("nodes", []):
        node["params"] = await _embed(node.get("params") or {}, read_media, files)
    if doc.get("params"):
        doc["params"] = await _embed(doc["params"], read_media, files)
    if mode == "baked":
        doc["baked"] = await _embed(doc.get("baked") or {}, read_media, files)
    else:
        doc["baked"] = {}
    if isinstance(meta.get("cover"), dict):
        meta["cover"] = await _embed(meta["cover"], read_media, files)
    doc["assets"] = [
        {"id": f"sha256:{digest}", "mime": mime, "bytes": len(data), "name": None}
        for digest, (data, mime) in sorted(files.items())
    ]
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("flow.json", json.dumps(doc, indent = 1))
        for digest, (data, mime) in sorted(files.items()):
            ext = asset_store.MIME_EXT.get(mime, "bin")
            zf.writestr(
                zipfile.ZipInfo(f"assets/{digest}.{ext}"), data, compress_type = zipfile.ZIP_STORED
            )
        cover = meta.get("cover")
        if _is_media(cover) and cover.get("mime") == "image/png":
            digest = asset_store.parse_ref(cover["ref"])
            if digest and digest in files:
                zf.writestr("preview.png", files[digest][0], compress_type = zipfile.ZIP_STORED)
    return out.getvalue(), f"{slug}.unslothflow", "application/zip"


def _read_entry(zf: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes:
    with zf.open(info) as fh:
        data = fh.read(limit + 1)
    if len(data) > limit:
        raise PackageError(f"{info.filename} is larger than allowed")
    return data


def _check_entries(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    infos = zf.infolist()
    if len(infos) > MAX_ENTRIES:
        raise PackageError("The package has too many files")
    seen: set[str] = set()
    total = 0
    for info in infos:
        name = info.filename
        if info.is_dir():
            if name.rstrip("/") not in ("assets",):
                raise PackageError(f"Unexpected folder {name!r} in the package")
            continue
        if (
            name.startswith(("/", "\\"))
            or "\\" in name
            or ".." in name.split("/")
            or re.match(r"^[A-Za-z]:", name)
        ):
            raise PackageError(f"Unsafe path {name!r} in the package")
        mode = info.external_attr >> 16
        if mode and stat.S_ISLNK(mode):
            raise PackageError(f"{name!r} is a link; packages cannot hold links")
        if name not in FIXED_ENTRIES and not _ASSET_NAME.match(name):
            raise PackageError(f"Unexpected file {name!r} in the package")
        folded = name.lower()
        if folded in seen:
            raise PackageError(f"{name!r} appears twice in the package")
        seen.add(folded)
        if info.file_size > schema.MAX_ASSET_BYTES:
            raise PackageError(
                f"{name!r} is larger than {schema.MAX_ASSET_BYTES // (1024 * 1024)} MB"
            )
        if (
            info.compress_size
            and info.file_size > 1024 * 1024
            and info.file_size / info.compress_size > MAX_RATIO
        ):
            raise PackageError(f"{name!r} expands suspiciously; refusing it")
        total += info.file_size
        if total > schema.MAX_PACKAGE_BYTES:
            raise PackageError("The package expands to more than 512 MB")
    if "flow.json" not in seen:
        raise PackageError("The package has no flow.json")
    return [i for i in infos if not i.is_dir()]


def _media_refs(value: Any) -> set[str]:
    found: set[str] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if _is_media(item) or _is_data_asset(item):
                digest = asset_store.parse_ref(item["ref"])
                if digest:
                    found.add(digest)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return found


def read_package(data: bytes) -> tuple[dict, dict[str, tuple[bytes, str]], dict]:
    """Parse and check a ``.json`` or ``.unslothflow`` upload. Returns (doc, assets by sha256, report); raises PackageError."""
    if len(data) > schema.MAX_PACKAGE_BYTES:
        raise PackageError("The file is larger than 512 MB")
    files: dict[str, tuple[bytes, str]] = {}
    warnings: list[str] = []
    if data[:4] == b"PK\x03\x04":
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            raise PackageError("The package is not a valid zip file") from None
        with zf:
            infos = _check_entries(zf)
            raw = None
            for info in infos:
                if info.filename == "flow.json":
                    raw = _read_entry(zf, info, schema.MAX_DOC_BYTES)
                    continue
                match = _ASSET_NAME.match(info.filename)
                if not match:
                    continue
                blob = _read_entry(zf, info, schema.MAX_ASSET_BYTES)
                digest, ext = match.groups()
                if hashlib.sha256(blob).hexdigest() != digest:
                    raise PackageError(f"{info.filename} does not match its checksum")
                mime = asset_store.sniff_mime(blob)
                if mime is None and ext == "json":
                    try:
                        json.loads(blob)
                    except (ValueError, RecursionError):
                        raise PackageError(f"{info.filename} is not valid JSON") from None
                    mime = "application/json"
                elif mime is None:
                    mime = asset_store.sniff_text(blob, asset_store.EXT_MIME.get(ext))
                if mime is None or asset_store.MIME_EXT.get(mime) != ext:
                    raise PackageError(f"{info.filename} is not the media type its name says")
                if mime == glb.GLB_MIME and (reason := glb.check(blob)):
                    raise PackageError(f"{info.filename}: {reason}")
                files[digest] = (blob, mime)
    else:
        raw = data
    doc, errors = parse_document(raw)
    if doc is None:
        raise PackageError("; ".join(e["message"] for e in errors[:5]) or "Not a valid flow")
    doc = scrub(share_flow(doc))
    declared = {asset_store.parse_ref(a["id"]) for a in doc.get("assets") or []}
    for digest in declared - set(files):
        warnings.append(f"Embedded file {digest[:12]} is missing from the package")
    used = (
        _media_refs(doc.get("nodes")) | _media_refs(doc.get("baked")) | _media_refs(doc.get("meta"))
    )
    missing = used - set(files) - {d for d in declared if d in files}
    if missing:
        warnings.append(f"{len(missing)} media file(s) the flow uses are not in the package")
    doc["assets"] = [a for a in doc.get("assets") or [] if asset_store.parse_ref(a["id"]) in files]
    for digest, (blob, mime) in files.items():
        if digest not in declared:
            doc["assets"].append(
                {"id": f"sha256:{digest}", "mime": mime, "bytes": len(blob), "name": None}
            )
    for group in doc.get("groups") or []:
        if group.get("kind") == "recipe" and (seed := missing_seed(group.get("config"))):
            warnings.append(f"{group.get('label') or 'Data Recipe'}: {seed}")
    unknown = sorted({n["type"] for n in doc.get("nodes", []) if n["type"] not in SPECS_BY_TYPE})
    if unknown:
        warnings.append(
            "Node types this Unsloth Studio does not know (the flow opens read-only until it is updated): "
            + ", ".join(unknown)
        )
    meta = doc.get("meta") or {}
    report = {
        "license": meta.get("license") if isinstance(meta.get("license"), str) else None,
        "content_rating": meta.get("content_rating")
        if isinstance(meta.get("content_rating"), str)
        else None,
        "author": meta.get("author") if isinstance(meta.get("author"), str) else None,
        "warnings": warnings,
        "baked_nodes": sorted((doc.get("baked") or {}).keys()),
        "unknown_types": unknown,
        "assets": len(files),
    }
    return doc, files, report


def store_assets(files: dict[str, tuple[bytes, str]]) -> None:
    for digest, (blob, mime) in files.items():
        stored, _ = asset_store.store(blob, mime)
        if stored != digest:
            raise PackageError("An asset changed while it was stored")
