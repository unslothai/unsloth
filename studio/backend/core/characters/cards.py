# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Character card import and export: Character Card V1, V2 and V3, as PNG, JSON or CHARX.

Specs: https://github.com/malfoyslastname/character-card-spec-v2 and
https://github.com/kwaroran/character-card-spec-v3. A PNG card carries base64 JSON in a
``chara`` (V1/V2) and/or ``ccv3`` (V3) text chunk; V3 is preferred when both are present.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
import re
import zipfile
import zlib
from dataclasses import dataclass, field
from typing import Any, Optional

from core.characters.placeholder import placeholder_png
from core.characters.png_text import (
    PNG_SIGNATURE,
    TEXT_CHUNKS,
    chunk_key,
    iter_png_chunks,
    png_chunk,
    read_text_chunk,
    strip_png_text,
)
from core.inference.mcp_images import MAX_IMAGE_PIXELS
from utils.image_files import sniff_image_type

MAX_CARD_BYTES = 40 * 1024 * 1024
MAX_AVATAR_BYTES = 20 * 1024 * 1024
MAX_FIELD_CHARS = 200_000
# Name, creator and version.
MAX_SHORT_CHARS = 200
MAX_GREETINGS = 64
MAX_TAGS = 64
MAX_TAG_CHARS = 100
# Every stored avatar is decoded once on import; this bounds what a browser will later decode.
MAX_AVATAR_PIXELS = MAX_IMAGE_PIXELS
_AVATAR_FORMATS = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}
_CARD_CHUNK_KEYS = ("ccv3", "chara")
_ARCHIVE_URIS = ("embeded://", "embedded://", "__asset:")
# Fields Studio edits; everything else in a card's data is carried through untouched.
EDITABLE_TEXT_FIELDS = (
    "name",
    "description",
    "personality",
    "scenario",
    "first_mes",
    "mes_example",
    "system_prompt",
    "post_history_instructions",
    "creator_notes",
    "creator",
    "character_version",
)


class CardError(ValueError):
    """The file is not a character card Studio can read."""


def _finite(text: str) -> Optional[float]:
    value = float(text)
    return value if math.isfinite(value) else None


_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def _well_formed(value: Any) -> Any:
    # A lone surrogate ("\ud835", half an emoji) parses but SQLite cannot store it.
    if isinstance(value, str):
        return _LONE_SURROGATE.sub("�", value)
    if isinstance(value, list):
        return [_well_formed(item) for item in value]
    if isinstance(value, dict):
        return {_well_formed(key): _well_formed(item) for key, item in value.items()}
    return value


def _json_loads(text: str):
    # NaN, Infinity and 1e309 read as null: they cannot be served back as JSON.
    try:
        return _well_formed(
            json.loads(text, parse_constant = lambda _constant: None, parse_float = _finite)
        )
    except json.JSONDecodeError:
        raise
    except (ValueError, RecursionError) as exc:
        # A 5000-digit integer or JSON nested thousands deep.
        raise CardError(
            "The card's JSON is nested too deeply or has a number too long to read."
        ) from exc


@dataclass
class ParsedCard:
    fields: dict[str, Any]
    card: dict[str, Any]
    avatar: Optional[bytes] = None
    avatar_type: Optional[str] = None
    source_format: str = "json"
    spec: str = "chara_card_v1"
    warnings: list[str] = field(default_factory = list)


def checked_avatar(data: Optional[bytes]) -> Optional[tuple[bytes, str]]:
    """The avatar as stored and its type: a PNG, JPEG, WebP or GIF that decodes within the pixel limit,
    PNG text chunks removed. None for anything else, including a decompression bomb."""
    media_type = sniff_image_type(data) if data else None
    if media_type is None:
        return None
    from PIL import Image

    if media_type == "image/png":
        # Before decoding: Pillow refuses a PNG whose compressed text passes its chunk limit.
        data = strip_png_text(data)
    try:
        with Image.open(io.BytesIO(data), formats = tuple(_AVATAR_FORMATS)) as image:
            width, height = image.size
            # Phone JPEGs with extra pictures (MPF) open as MPO, which is not an opener of its own.
            found = "JPEG" if image.format == "MPO" else image.format
            if _AVATAR_FORMATS.get(found) != media_type or not (
                0 < width * height <= MAX_AVATAR_PIXELS
            ):
                return None
            image.load()
    except Exception:
        return None
    return data, media_type


def character_fields(parsed: ParsedCard) -> dict:
    return {camel(key): value for key, value in parsed.fields.items()}


def _decode_card_text(raw: bytes) -> Optional[dict]:
    raw = raw.strip()
    # Base64 may be line-wrapped; padding is counted without the whitespace.
    compact = b"".join(raw.split())
    candidates = []
    try:
        candidates.append(base64.b64decode(compact + b"=" * (-len(compact) % 4), validate = False))
    except (binascii.Error, ValueError):
        pass
    # Raw JSON: tEXt bytes are usually UTF-8 despite the spec's latin-1.
    candidates.append(raw)
    for candidate in candidates:
        for encoding in ("utf-8", "latin-1"):
            try:
                value = _json_loads(candidate.decode(encoding))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                return value
    return None


def read_png_card(data: bytes) -> tuple[Optional[dict], list[str]]:
    """The card JSON in a PNG (ccv3 over chara) and which card chunks it held."""
    if not data.startswith(PNG_SIGNATURE):
        raise CardError("Not a PNG file.")
    found: dict[str, dict] = {}
    budget = [MAX_CARD_BYTES]
    for kind, body, _start, _end in iter_png_chunks(data):
        if kind not in TEXT_CHUNKS:
            continue
        wanted = tuple(key for key in _CARD_CHUNK_KEYS if key not in found)
        if not wanted:
            break
        parsed = read_text_chunk(kind, body, wanted, budget)
        if parsed is None:
            continue
        card = _decode_card_text(parsed[1])
        if card is not None:
            found[parsed[0]] = card
    for key in _CARD_CHUNK_KEYS:
        if key in found:
            return found[key], sorted(found)
    return None, sorted(found)


def embed_card_in_png(png: bytes, chunks: dict[str, dict]) -> bytes:
    """Writes each card as a base64 tEXt chunk, replacing any card chunks already there."""
    out = bytearray(PNG_SIGNATURE)
    for kind, body, start, end in iter_png_chunks(png):
        if kind == b"IEND":
            for key, card in chunks.items():
                encoded = base64.b64encode(json.dumps(card, ensure_ascii = False).encode("utf-8"))
                out.extend(png_chunk(b"tEXt", key.encode("latin-1") + b"\x00" + encoded))
            out.extend(png[start:end])
            return bytes(out)
        if kind in TEXT_CHUNKS and chunk_key(body) in _CARD_CHUNK_KEYS:
            continue
        out.extend(png[start:end])
    raise CardError("The avatar PNG has no end marker.")


def to_png(image: Optional[bytes], name: str = "") -> bytes:
    """A PNG of the avatar, or a plain placeholder when there is none or it cannot be decoded."""
    if image and image.startswith(PNG_SIGNATURE):
        return strip_png_text(image)
    if image:
        from PIL import Image, ImageOps
        try:
            with Image.open(io.BytesIO(image)) as source:
                source.load()
                # Browsers show a phone JPEG by its EXIF orientation; the exported PNG has none.
                upright = ImageOps.exif_transpose(source) or source
                alpha = "A" in upright.getbands() or "transparency" in upright.info
                converted = upright.convert("RGBA" if alpha else "RGB")
                buffer = io.BytesIO()
                converted.save(buffer, format = "PNG")
                return buffer.getvalue()
        except Exception:
            pass
    return placeholder_png(name)


def read_charx(data: bytes) -> tuple[dict, Optional[bytes]]:
    try:
        return _read_charx(data)
    except CardError:
        raise
    except (
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        RuntimeError,
        NotImplementedError,
        EOFError,
        KeyError,
        zlib.error,
        ValueError,
        OSError,
    ) as exc:
        # Encrypted entries, unknown compression, or a damaged directory (negative seek, non-UTF-8 name).
        raise CardError("This CHARX archive could not be read.") from exc


def _icon_uris(card: dict) -> list[str]:
    """Archive paths of the card's icon assets, the main one first."""
    data = card.get("data")
    assets = data.get("assets") if isinstance(data, dict) else None
    if not isinstance(assets, list):
        return []
    icons = [a for a in assets if isinstance(a, dict) and a.get("type") == "icon"]
    icons.sort(key = lambda a: 0 if a.get("name") == "main" else 1)
    paths = []
    for asset in icons:
        uri = str(asset.get("uri") or "")
        for prefix in _ARCHIVE_URIS:
            if uri.startswith(prefix):
                paths.append(uri[len(prefix) :])
    return paths


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    # Streamed to the declared size: ZipFile.read inflates the whole member before it checks that size.
    with archive.open(info) as member:
        return member.read(info.file_size)


def _read_charx(data: bytes) -> tuple[dict, Optional[bytes]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise CardError("Not a CHARX archive.") from exc
    with archive:
        names = {info.filename: info for info in archive.infolist()}
        if "card.json" not in names:
            raise CardError("The CHARX archive has no card.json.")
        if names["card.json"].file_size > MAX_CARD_BYTES:
            raise CardError("card.json is too large.")
        try:
            card = _json_loads(_read_member(archive, names["card.json"]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CardError("card.json is not valid JSON.") from exc
        if not isinstance(card, dict):
            raise CardError("card.json is not a character card.")
        icon_bytes = None
        candidates = _icon_uris(card)
        candidates.extend(sorted(n for n in names if n.startswith("assets/icon/")))
        # Each member is read once, and all reads together stay within one card's size.
        budget = MAX_CARD_BYTES
        for candidate in dict.fromkeys(candidates):
            info = names.get(candidate)
            if info is not None and 0 < info.file_size <= budget:
                budget -= info.file_size
                checked = checked_avatar(_read_member(archive, info))
                if checked:
                    icon_bytes = checked[0]
                    break
        return card, icon_bytes


def _text(value: Any, limit: Optional[int] = MAX_FIELD_CHARS) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.replace("\r\n", "\n")[:limit]


def _clip(text: str, limit: int, label: str, warnings: Optional[list[str]]) -> str:
    if len(text) <= limit:
        return text
    if warnings is not None:
        warnings.append(f"Shortened the {label} to {limit:,} characters.")
    return text[:limit]


def _string_list(
    value: Any,
    limit: int,
    max_chars: int = MAX_FIELD_CHARS,
    warnings: Optional[list[str]] = None,
    label: str = "items",
) -> list[str]:
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    clipped = False
    for item in value:
        text = _text(item, None).strip() if isinstance(item, str) else ""
        if len(text) > max_chars:
            clipped = True
            text = text[:max_chars].strip()
        if text and text not in seen:
            seen.add(text)
            if len(out) < limit:
                out.append(text)
    if warnings is not None:
        if len(seen) > limit:
            warnings.append(f"Kept the first {limit} of {len(seen)} {label}.")
        if clipped:
            warnings.append(f"Shortened {label} longer than {max_chars:,} characters.")
    return out


def card_data(card: dict) -> tuple[dict, str]:
    """The card's data object and its spec; a V1 card is its own data."""
    spec = str(card.get("spec") or "")
    data = card.get("data")
    if spec in ("chara_card_v2", "chara_card_v3") and isinstance(data, dict):
        return data, spec
    if isinstance(data, dict) and "name" in data and "name" not in card:
        return data, spec or "chara_card_v2"
    return card, "chara_card_v1"


_CARD_TEXT_KEYS = (
    "description",
    "personality",
    "scenario",
    "first_mes",
    "mes_example",
    "char_persona",
    "char_greeting",
    "example_dialogue",
    "world_scenario",
)


def normalize_card(
    card: dict,
    fallback_name: str = "",
    warnings: Optional[list[str]] = None,
) -> dict[str, Any]:
    """The editable fields; anything cut to fit a limit is noted in *warnings*."""
    if not isinstance(card, dict):
        raise CardError("Not a character card.")
    data, _spec = card_data(card)
    # A named lorebook would otherwise import as an empty character.
    if "entries" in card and data is card and not any(card.get(key) for key in _CARD_TEXT_KEYS):
        raise CardError("This file is a lorebook, not a character card.")

    def fit(
        value: Any,
        label: str,
        limit: int = MAX_FIELD_CHARS,
    ) -> str:
        return _clip(_text(value, None), limit, label, warnings)

    name = _text(data.get("name") or card.get("name") or card.get("char_name"), None).strip()
    if not name:
        if "entries" in card:
            raise CardError("This file is a lorebook, not a character card.")
        if isinstance(card.get("node"), dict):
            raise CardError(
                "This is a website's API response, not a character card. Download the card's PNG instead."
            )
        if data is card or not fallback_name.strip():
            raise CardError("The card has no name.")
        name = fallback_name.strip()
    if (
        data is card
        and not any(_text(card.get(key)).strip() for key in _CARD_TEXT_KEYS)
        and not card.get("alternate_greetings")
        and not isinstance(card.get("character_book"), dict)
    ):
        raise CardError(
            "This file is not a character card: it has a name but no description, personality, scenario or greeting."
        )
    extensions = data.get("extensions") if isinstance(data.get("extensions"), dict) else {}
    talkativeness = extensions.get("talkativeness", card.get("talkativeness", 0.5))
    try:
        talkativeness = float(talkativeness)
        talkativeness = min(1.0, max(0.0, talkativeness)) if math.isfinite(talkativeness) else 0.5
    except (TypeError, ValueError):
        talkativeness = 0.5
    depth_prompt = extensions.get("depth_prompt")
    if isinstance(depth_prompt, dict) and _text(depth_prompt.get("prompt")).strip():
        try:
            depth = max(0, min(100, int(depth_prompt.get("depth", 4))))
        except (TypeError, ValueError):
            depth = 4
        depth_prompt = {
            "prompt": fit(depth_prompt.get("prompt"), "depth prompt"),
            "depth": depth,
            "role": depth_prompt.get("role")
            if depth_prompt.get("role") in ("system", "user", "assistant")
            else "system",
        }
    else:
        depth_prompt = None
    book = data.get("character_book")
    greetings = [
        _text(g, None)
        for g in _as_list(data.get("alternate_greetings"))
        if isinstance(g, str) and g.strip()
    ]
    if len(greetings) > MAX_GREETINGS and warnings is not None:
        warnings.append(f"Kept the first {MAX_GREETINGS} of {len(greetings)} alternate greetings.")
    greetings = greetings[:MAX_GREETINGS]
    if any(len(g) > MAX_FIELD_CHARS for g in greetings):
        greetings = [g[:MAX_FIELD_CHARS] for g in greetings]
        if warnings is not None:
            warnings.append(
                f"Shortened alternate greetings longer than {MAX_FIELD_CHARS:,} characters."
            )
    fields = {
        "name": _clip(name, MAX_SHORT_CHARS, "name", warnings),
        "description": fit(data.get("description") or card.get("char_persona"), "description"),
        "personality": fit(data.get("personality"), "personality"),
        "scenario": fit(data.get("scenario") or card.get("world_scenario"), "scenario"),
        "first_mes": fit(data.get("first_mes") or card.get("char_greeting"), "greeting"),
        "mes_example": fit(
            data.get("mes_example") or card.get("example_dialogue"), "example messages"
        ),
        "system_prompt": fit(data.get("system_prompt"), "system prompt"),
        "post_history_instructions": fit(
            data.get("post_history_instructions"), "post-history instructions"
        ),
        "creator_notes": fit(
            data.get("creator_notes") or card.get("creatorcomment"), "creator notes"
        ),
        "creator": fit(data.get("creator"), "creator", MAX_SHORT_CHARS),
        "character_version": fit(data.get("character_version"), "version", MAX_SHORT_CHARS),
        "alternate_greetings": greetings,
        # MAX_TAG_CHARS matches the editor's limit, so an imported card stays editable.
        "tags": _string_list(
            data.get("tags") if "tags" in data else card.get("tags"),
            MAX_TAGS,
            MAX_TAG_CHARS,
            warnings,
            "tags",
        ),
        "talkativeness": talkativeness,
        "favorite": bool(extensions.get("fav") or card.get("fav")),
        "depth_prompt": depth_prompt,
        "character_book": normalize_book(book) if isinstance(book, dict) else None,
        "extensions": extensions,
    }
    return fields


_SELECTIVE_LOGIC = {0: "and_any", 1: "not_all", 2: "not_any", 3: "and_all"}
_SELECTIVE_LOGIC_CODE = {name: code for code, name in _SELECTIVE_LOGIC.items()}


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, str) else []


def normalize_book(book: dict) -> dict:
    """Keeps the book as the spec shapes it, with entries made consistent for the prompt builder."""
    entries_in = book.get("entries")
    if isinstance(entries_in, dict):  # SillyTavern world files key entries by uid.
        entries_in = list(entries_in.values())
    entries: list[dict] = []
    for index, entry in enumerate(entries_in if isinstance(entries_in, list) else []):
        if not isinstance(entry, dict):
            continue
        ext = entry.get("extensions") if isinstance(entry.get("extensions"), dict) else {}
        raw_position = entry.get("position")
        position = raw_position
        if position not in ("before_char", "after_char"):
            position = "after_char" if position in (1, "1") else "before_char"
        # SillyTavern's placement: 0 before, 1 after the character, 4 at a depth. Cards keep it in
        # extensions, world files at the top level.
        ext_position = ext.get("position", raw_position if type(raw_position) is int else None)
        if ext_position == 4 or raw_position == "at_depth":
            position = "at_depth"
        elif ext_position in (0, 1):
            position = "before_char" if ext_position == 0 else "after_char"
        logic = entry.get("selective_logic")
        if not (isinstance(logic, str) and logic in _SELECTIVE_LOGIC_CODE):
            code = ext.get("selectiveLogic", entry.get("selectiveLogic"))
            if isinstance(code, str) and code in _SELECTIVE_LOGIC_CODE:
                logic = code
            else:
                logic = _SELECTIVE_LOGIC.get(code, "and_any") if type(code) is int else "and_any"
        case_sensitive = entry.get("case_sensitive")
        if not isinstance(case_sensitive, bool):
            case_sensitive = bool(ext.get("case_sensitive"))
        enabled = entry.get("enabled")
        if not isinstance(enabled, (bool, int)):
            enabled = not entry.get("disable", False)
        normalized = {
            **entry,
            "id": entry.get("id", entry.get("uid", index)),
            "keys": _string_list(entry.get("keys", entry.get("key")), 100),
            "secondary_keys": _string_list(
                entry.get("secondary_keys", entry.get("keysecondary")), 100
            ),
            "content": _text(entry.get("content")),
            "comment": _text(entry.get("comment") or entry.get("name"))[:500],
            "enabled": bool(enabled),
            "constant": bool(entry.get("constant")),
            "selective": bool(entry.get("selective")),
            "case_sensitive": case_sensitive,
            "insertion_order": _int(entry.get("insertion_order", entry.get("order")), 100),
            "position": position,
            "depth": _int(ext.get("depth", entry.get("depth")), 4),
            "selective_logic": logic,
            "extensions": ext,
        }
        entries.append(normalized)
    return {
        **book,
        "name": _text(book.get("name"))[:200],
        "scan_depth": _int(book.get("scan_depth"), 4),
        "token_budget": _int(book.get("token_budget"), 1024),
        "recursive_scanning": bool(book.get("recursive_scanning")),
        "extensions": book.get("extensions") if isinstance(book.get("extensions"), dict) else {},
        "entries": entries,
    }


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_card_file(data: bytes, filename: str = "") -> ParsedCard:
    if not data:
        raise CardError("The file is empty.")
    if len(data) > MAX_CARD_BYTES:
        raise CardError("The file is too large for a character card.")
    avatar = None
    if data.startswith(PNG_SIGNATURE):
        card, chunks = read_png_card(data)
        if card is None:
            raise CardError("This PNG has no character card inside it.")
        checked = checked_avatar(data)
        avatar = checked[0] if checked else None
        source_format = "png"
        warnings = [] if checked else ["The picture could not be read, so the character has none."]
        if chunks == ["ccv3", "chara"]:
            warnings.append("Read the V3 data; the file also carries V2 data for older apps.")
    elif data[:2] == b"PK" or zipfile.is_zipfile(io.BytesIO(data)):
        # RisuAI writes CHARX with a JPEG cover in front of the zip, which zipfile still opens.
        card, avatar = read_charx(data)
        source_format = "charx"
        warnings = []
        if avatar is None and checked_avatar(data):
            # No icon asset: the cover image in front of the archive is the avatar.
            avatar = to_png(data)
        if avatar is None and _icon_uris(card):
            warnings.append("The picture could not be read, so the character has none.")
    else:
        try:
            text = data.decode("utf-8-sig")
            card = _json_loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if sniff_image_type(data):
                raise CardError("This image has no character card inside it.") from exc
            raise CardError("The file is not a PNG, JSON or CHARX character card.") from exc
        if not isinstance(card, dict):
            raise CardError("The JSON is not a character card.")
        source_format = "json"
        warnings = []
    if avatar is not None and len(avatar) > MAX_AVATAR_BYTES:
        avatar = None
        warnings.append(
            f"The picture is larger than {MAX_AVATAR_BYTES // (1024 * 1024)} MB, so the character has none."
        )
    fields = normalize_card(card, re.split(r"[\\/]", filename)[-1].rsplit(".", 1)[0], warnings)
    _data, spec = card_data(card)
    return ParsedCard(
        fields = fields,
        card = card,
        avatar = avatar,
        avatar_type = sniff_image_type(avatar) if avatar else None,
        source_format = source_format,
        spec = spec,
        warnings = warnings,
    )


def _base_data(character: dict) -> dict:
    stored = character.get("card")
    if isinstance(stored, dict):
        data, spec = card_data(stored)
        # A V1 card's other top-level keys (avatar, chat, create_date...) are not V2 data.
        if spec != "chara_card_v1":
            return json.loads(json.dumps(data))
    return {}


def _overlay(character: dict, data: dict) -> dict:
    for key in EDITABLE_TEXT_FIELDS:
        data[key] = character.get(camel(key), "") or ""
    data["alternate_greetings"] = list(character.get("alternateGreetings") or [])
    data["tags"] = list(character.get("tags") or [])
    extensions = dict(data["extensions"]) if isinstance(data.get("extensions"), dict) else {}
    extensions.update(character.get("extensions") or {})
    extensions["talkativeness"] = character.get("talkativeness", 0.5)
    extensions["fav"] = bool(character.get("favorite"))
    if character.get("depthPrompt"):
        extensions["depth_prompt"] = character["depthPrompt"]
    else:
        extensions.pop("depth_prompt", None)
    data["extensions"] = extensions
    book = character.get("characterBook")
    if book:
        data["character_book"] = _export_book(book)
    else:
        data.pop("character_book", None)
    return data


_EXT_POSITION = {"before_char": 0, "after_char": 1, "at_depth": 4}


def _export_book(book: dict) -> dict:
    """The book with each entry's edits also in SillyTavern's extensions, which importers read first."""
    entries = []
    for entry in book.get("entries") or []:
        if not isinstance(entry, dict):
            entries.append(entry)
            continue
        ext = dict(entry["extensions"]) if isinstance(entry.get("extensions"), dict) else {}
        position = entry.get("position")
        # Placements Studio does not model (2, 3, 5, 6) stay as imported unless the entry is at a depth.
        if position == "at_depth" or ext.get("position") in (0, 1, 4, None):
            if position in _EXT_POSITION:
                ext["position"] = _EXT_POSITION[position]
        if "depth" in entry and (position == "at_depth" or "depth" in ext):
            ext["depth"] = entry["depth"]
        logic = entry.get("selective_logic")
        if (
            isinstance(logic, str)
            and logic in _SELECTIVE_LOGIC_CODE
            and (logic != "and_any" or "selectiveLogic" in ext)
        ):
            ext["selectiveLogic"] = _SELECTIVE_LOGIC_CODE[logic]
        if isinstance(entry.get("case_sensitive"), bool) and (
            entry["case_sensitive"] or "case_sensitive" in ext
        ):
            ext["case_sensitive"] = entry["case_sensitive"]
        out = {**entry, "extensions": ext}
        if position == "at_depth":
            # Not a spec value; extensions.position 4 carries the depth placement.
            out["position"] = "before_char"
        entries.append(out)
    return {**book, "entries": entries}


def camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(part.capitalize() for part in rest)


def build_card_v2(character: dict) -> dict:
    data = _overlay(character, _base_data(character))
    for v3_only in (
        "assets",
        "nickname",
        "creator_notes_multilingual",
        "source",
        "group_only_greetings",
        "creation_date",
        "modification_date",
    ):
        data.pop(v3_only, None)
    return {"spec": "chara_card_v2", "spec_version": "2.0", "data": data}


def build_card_v3(character: dict) -> dict:
    data = _overlay(character, _base_data(character))
    data.setdefault("group_only_greetings", [])
    # The exported file is the main icon; CHARX-embedded assets are not carried.
    kept = [
        asset
        for asset in (data.get("assets") if isinstance(data.get("assets"), list) else [])
        if isinstance(asset, dict)
        and not str(asset.get("uri") or "").startswith(_ARCHIVE_URIS)
        and not (asset.get("type") == "icon" and asset.get("name") == "main")
    ]
    data["assets"] = [{"type": "icon", "uri": "ccdefault:", "name": "main", "ext": "png"}, *kept]
    if character.get("createdAt"):
        data.setdefault("creation_date", int(character["createdAt"]) // 1000)
    if character.get("updatedAt"):
        data["modification_date"] = int(character["updatedAt"]) // 1000
    return {"spec": "chara_card_v3", "spec_version": "3.0", "data": data}


def export_png(character: dict, avatar: Optional[bytes]) -> bytes:
    chunks = {"chara": build_card_v2(character), "ccv3": build_card_v3(character)}
    try:
        return embed_card_in_png(to_png(avatar, character.get("name") or ""), chunks)
    except CardError:
        # A damaged avatar (a truncated PNG) still exports, on a plain placeholder.
        return embed_card_in_png(to_png(None, character.get("name") or ""), chunks)
