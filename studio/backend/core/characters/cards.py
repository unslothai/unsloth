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
import struct
import zipfile
import zlib
from dataclasses import dataclass, field
from typing import Any, Optional

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_CARD_BYTES = 40 * 1024 * 1024
MAX_FIELD_CHARS = 200_000
MAX_GREETINGS = 64
MAX_TAGS = 64
MAX_TAG_CHARS = 100
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


def _json_loads(text: str):
    # NaN, Infinity and overflowing numbers (1e309) parse in Python but cannot be served back as JSON; read them as null.
    return json.loads(text, parse_constant = lambda _constant: None, parse_float = _finite)


@dataclass
class ParsedCard:
    fields: dict[str, Any]
    card: dict[str, Any]
    avatar: Optional[bytes] = None
    avatar_type: Optional[str] = None
    source_format: str = "json"
    spec: str = "chara_card_v1"
    warnings: list[str] = field(default_factory = list)


def sniff_image_type(data: bytes) -> Optional[str]:
    if data.startswith(PNG_SIGNATURE):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return None


def character_fields(parsed: "ParsedCard") -> dict:
    """A parsed card as the fields a stored character takes."""
    fields = parsed.fields
    return {
        "name": fields["name"],
        "description": fields["description"],
        "personality": fields["personality"],
        "scenario": fields["scenario"],
        "firstMes": fields["first_mes"],
        "alternateGreetings": fields["alternate_greetings"],
        "mesExample": fields["mes_example"],
        "systemPrompt": fields["system_prompt"],
        "postHistoryInstructions": fields["post_history_instructions"],
        "creatorNotes": fields["creator_notes"],
        "creator": fields["creator"],
        "characterVersion": fields["character_version"],
        "tags": fields["tags"],
        "talkativeness": fields["talkativeness"],
        "favorite": fields["favorite"],
        "depthPrompt": fields["depth_prompt"],
        "characterBook": fields["character_book"],
        "extensions": fields["extensions"],
    }


def _iter_png_chunks(data: bytes):
    if not data.startswith(PNG_SIGNATURE):
        raise CardError("Not a PNG file.")
    offset = len(PNG_SIGNATURE)
    while offset + 8 <= len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        kind = data[offset + 4 : offset + 8]
        start = offset + 8
        end = start + length
        if end + 4 > len(data):
            # Truncated file: stop at what is readable rather than fail the whole card.
            return
        yield kind, data[start:end], offset, end + 4
        offset = end + 4
        if kind == b"IEND":
            return


def _inflate(data: bytes, budget: list[int]) -> bytes:
    # Bounded across the whole file: a few KB of zlib can expand to gigabytes, and a PNG can repeat chunks.
    inflater = zlib.decompressobj()
    out = inflater.decompress(data, max(budget[0], 1))
    budget[0] -= len(out)
    if inflater.unconsumed_tail or not inflater.eof or budget[0] < 0:
        raise ValueError("compressed text chunk too large or truncated")
    return out


def _png_text_chunk(
    kind: bytes,
    body: bytes,
    keys: tuple[str, ...] = _CARD_CHUNK_KEYS,
    budget: Optional[list[int]] = None,
) -> Optional[tuple[str, str]]:
    """A text chunk's key and text; text is decoded only for the wanted keys, so other chunks cost nothing."""
    try:
        key, _, rest = body.partition(b"\x00")
        key = key.decode("latin-1")
        if key.lower() not in keys:
            return key, ""
        budget = [MAX_CARD_BYTES] if budget is None else budget
        if kind == b"tEXt":
            return key, rest.decode("latin-1")
        if kind == b"zTXt":
            return key, _inflate(rest[1:], budget).decode("latin-1")
        if kind == b"iTXt":
            compressed, rest = rest[0], rest[2:]
            _language, _, rest = rest.partition(b"\x00")
            _translated, _, text = rest.partition(b"\x00")
            if compressed:
                text = _inflate(text, budget)
            return key, text.decode("utf-8")
    except (zlib.error, UnicodeDecodeError, IndexError, ValueError):
        return None
    return None


def _decode_card_text(text: str) -> Optional[dict]:
    text = text.strip()
    candidates = []
    try:
        candidates.append(base64.b64decode(text + "=" * (-len(text) % 4), validate = False))
    except (binascii.Error, ValueError):
        pass
    candidates.append(text.encode("utf-8", "surrogatepass"))
    for raw in candidates:
        for encoding in ("utf-8", "latin-1"):
            try:
                value = _json_loads(raw.decode(encoding))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                continue
            if isinstance(value, dict):
                return value
    return None


def read_png_card(data: bytes) -> tuple[Optional[dict], list[str]]:
    """The card JSON in a PNG (ccv3 over chara) and which card chunks it held."""
    found: dict[str, dict] = {}
    budget = [MAX_CARD_BYTES]
    for kind, body, _start, _end in _iter_png_chunks(data):
        if kind not in (b"tEXt", b"zTXt", b"iTXt"):
            continue
        wanted = tuple(key for key in _CARD_CHUNK_KEYS if key not in found)
        if not wanted:
            break
        parsed = _png_text_chunk(kind, body, wanted, budget)
        if parsed is None:
            continue
        key, text = parsed
        key = key.lower()
        if key in _CARD_CHUNK_KEYS and key not in found:
            card = _decode_card_text(text)
            if card is not None:
                found[key] = card
    for key in _CARD_CHUNK_KEYS:
        if key in found:
            return found[key], sorted(found)
    return None, sorted(found)


def strip_png_text(data: bytes) -> bytes:
    """The PNG without its text chunks, so a stored portrait does not carry a stale card."""
    out = bytearray(PNG_SIGNATURE)
    for kind, _body, start, end in _iter_png_chunks(data):
        if kind in (b"tEXt", b"zTXt", b"iTXt"):
            continue
        out.extend(data[start:end])
    return bytes(out)


def _png_chunk(kind: bytes, body: bytes) -> bytes:
    crc = zlib.crc32(kind + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)


def embed_card_in_png(png: bytes, chunks: dict[str, dict]) -> bytes:
    """Writes each card as a base64 tEXt chunk, replacing any card chunks already there."""
    out = bytearray(PNG_SIGNATURE)
    for kind, body, start, end in _iter_png_chunks(png):
        if kind == b"IEND":
            for key, card in chunks.items():
                encoded = base64.b64encode(json.dumps(card, ensure_ascii = False).encode("utf-8"))
                out.extend(_png_chunk(b"tEXt", key.encode("latin-1") + b"\x00" + encoded))
            out.extend(png[start:end])
            return bytes(out)
        if kind in (b"tEXt", b"zTXt", b"iTXt"):
            parsed = _png_text_chunk(kind, body, ())
            if parsed is not None and parsed[0].lower() in _CARD_CHUNK_KEYS:
                continue
        out.extend(png[start:end])
    raise CardError("The portrait PNG has no end marker.")


def to_png(image: Optional[bytes], name: str = "") -> bytes:
    """A PNG of the portrait, or a plain placeholder when there is none or it cannot be decoded."""
    from PIL import Image

    if image and image.startswith(PNG_SIGNATURE):
        return strip_png_text(image)
    if image:
        try:
            with Image.open(io.BytesIO(image)) as source:
                source.load()
                converted = source.convert("RGBA" if "A" in source.getbands() else "RGB")
                buffer = io.BytesIO()
                converted.save(buffer, format = "PNG")
                return buffer.getvalue()
        except Exception:
            pass
    seed = sum(name.encode("utf-8")) % 360 if name else 210
    placeholder = Image.new("RGB", (400, 600), _hsl_to_rgb(seed, 0.35, 0.42))
    buffer = io.BytesIO()
    placeholder.save(buffer, format = "PNG")
    return buffer.getvalue()


def _hsl_to_rgb(h: float, s: float, light: float) -> tuple[int, int, int]:
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h / 360, light, s)
    return int(r * 255), int(g * 255), int(b * 255)


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
    ) as exc:
        # Encrypted entries, unknown compression or a damaged archive.
        raise CardError("This CHARX archive could not be read.") from exc


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
        assets = (
            (card.get("data") or {}).get("assets") if isinstance(card.get("data"), dict) else None
        )
        candidates: list[str] = []
        if isinstance(assets, list):
            icons = [a for a in assets if isinstance(a, dict) and a.get("type") == "icon"]
            icons.sort(key = lambda a: 0 if a.get("name") == "main" else 1)
            for asset in icons:
                uri = str(asset.get("uri") or "")
                for prefix in _ARCHIVE_URIS:
                    if uri.startswith(prefix):
                        candidates.append(uri[len(prefix) :])
        candidates.extend(sorted(n for n in names if n.startswith("assets/icon/")))
        # Each member is read once, and all reads together stay within one card's size.
        budget = MAX_CARD_BYTES
        for candidate in dict.fromkeys(candidates):
            info = names.get(candidate)
            if info is not None and 0 < info.file_size <= budget:
                budget -= info.file_size
                blob = _read_member(archive, info)
                if sniff_image_type(blob):
                    icon_bytes = blob
                    break
        return card, icon_bytes


def _text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.replace("\r\n", "\n")[:MAX_FIELD_CHARS]


def _string_list(
    value: Any,
    limit: int,
    max_chars: int = MAX_FIELD_CHARS,
) -> list[str]:
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _text(item)[:max_chars].strip() if isinstance(item, str) else ""
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
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


def normalize_card(card: dict) -> dict[str, Any]:
    if not isinstance(card, dict):
        raise CardError("Not a character card.")
    data, _spec = card_data(card)
    # A lorebook has its entries at the top and none of a card's text; one with a name would
    # otherwise import as an empty character.
    if "entries" in card and data is card and not any(card.get(key) for key in _CARD_TEXT_KEYS):
        raise CardError("This file is a lorebook, not a character card.")
    name = _text(data.get("name") or card.get("name") or card.get("char_name")).strip()
    if not name:
        if "entries" in card:
            raise CardError("This file is a lorebook, not a character card.")
        if isinstance(card.get("node"), dict):
            raise CardError(
                "This is a website's API response, not a character card. Download the card's PNG instead."
            )
        raise CardError("The card has no name.")
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
            "prompt": _text(depth_prompt.get("prompt")),
            "depth": depth,
            "role": depth_prompt.get("role")
            if depth_prompt.get("role") in ("system", "user", "assistant")
            else "system",
        }
    else:
        depth_prompt = None
    book = data.get("character_book")
    fields = {
        "name": name[:200],
        "description": _text(data.get("description") or card.get("char_persona")),
        "personality": _text(data.get("personality")),
        "scenario": _text(data.get("scenario") or card.get("world_scenario")),
        "first_mes": _text(data.get("first_mes") or card.get("char_greeting")),
        "mes_example": _text(data.get("mes_example") or card.get("example_dialogue")),
        "system_prompt": _text(data.get("system_prompt")),
        "post_history_instructions": _text(data.get("post_history_instructions")),
        "creator_notes": _text(data.get("creator_notes") or card.get("creatorcomment")),
        "creator": _text(data.get("creator"))[:200],
        "character_version": _text(data.get("character_version"))[:100],
        "alternate_greetings": [
            _text(g)
            for g in _as_list(data.get("alternate_greetings"))
            if isinstance(g, str) and g.strip()
        ][:MAX_GREETINGS],
        # MAX_TAG_CHARS matches the editor's limit, so an imported card stays editable.
        "tags": _string_list(
            data.get("tags") if "tags" in data else card.get("tags"), MAX_TAGS, MAX_TAG_CHARS
        ),
        "talkativeness": talkativeness,
        "favorite": bool(extensions.get("fav") or card.get("fav")),
        "depth_prompt": depth_prompt,
        "character_book": normalize_book(book) if isinstance(book, dict) else None,
        "extensions": extensions,
    }
    return fields


_LOGIC = {0: "and_any", 1: "not_all", 2: "not_any", 3: "and_all"}


def _as_list(value: Any) -> list:
    """A list as-is, a lone string as one item, anything else as nothing."""
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
        position = entry.get("position")
        if position not in ("before_char", "after_char"):
            position = "after_char" if position in (1, "1") else "before_char"
        # SillyTavern's own placement (extensions.position): 0 before, 1 after the character, 4 at a depth.
        ext_position = ext.get("position")
        if ext_position == 4 or entry.get("position") == "at_depth":
            position = "at_depth"
        elif ext_position in (0, 1):
            position = "before_char" if ext_position == 0 else "after_char"
        logic = ext.get("selectiveLogic", 0)
        if not isinstance(logic, (int, str)) or isinstance(logic, bool):
            logic = 0
        normalized = {
            **entry,
            "id": entry.get("id", entry.get("uid", index)),
            "keys": _string_list(entry.get("keys", entry.get("key")), 100),
            "secondary_keys": _string_list(
                entry.get("secondary_keys", entry.get("keysecondary")), 100
            ),
            "content": _text(entry.get("content")),
            "comment": _text(entry.get("comment") or entry.get("name"))[:500],
            "enabled": entry.get("enabled", not entry.get("disable", False)) is not False,
            "constant": bool(entry.get("constant")),
            "selective": bool(entry.get("selective")),
            "case_sensitive": bool(entry.get("case_sensitive") or ext.get("case_sensitive")),
            "insertion_order": _int(entry.get("insertion_order", entry.get("order")), 100),
            "position": position,
            "depth": _int(ext.get("depth", entry.get("depth")), 4),
            "selective_logic": _LOGIC.get(logic, logic if logic in _LOGIC.values() else "and_any"),
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
    lower = filename.lower()
    avatar = None
    if data.startswith(PNG_SIGNATURE):
        card, chunks = read_png_card(data)
        if card is None:
            raise CardError("This PNG has no character card inside it.")
        avatar = strip_png_text(data)
        source_format = "png"
        warnings = []
        if chunks == ["ccv3", "chara"]:
            warnings.append("Read the V3 data; the file also carries V2 data for older apps.")
    elif data[:2] == b"PK" or lower.endswith(".charx") or zipfile.is_zipfile(io.BytesIO(data)):
        # RisuAI writes CHARX with a JPEG cover in front of the zip, which zipfile still opens.
        card, avatar = read_charx(data)
        source_format = "charx"
        warnings = []
        if avatar is None and sniff_image_type(data):
            # No icon asset: the cover image in front of the archive is the portrait.
            avatar = to_png(data)
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
    fields = normalize_card(card)
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
        data, _spec = card_data(stored)
        return json.loads(json.dumps(data))
    return {}


def _overlay(character: dict, data: dict) -> dict:
    for key in EDITABLE_TEXT_FIELDS:
        data[key] = character.get(_camel(key), "") or ""
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
    # A V1 card stored top-level keys that are not V2 data.
    for legacy in (
        "char_name",
        "char_persona",
        "char_greeting",
        "world_scenario",
        "example_dialogue",
        "spec",
        "spec_version",
        "data",
    ):
        data.pop(legacy, None)
    return data


_EXT_POSITION = {"before_char": 0, "after_char": 1, "at_depth": 4}


def _export_book(book: dict) -> dict:
    """The book with each entry's placement also in SillyTavern's extensions, which importers read first."""
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
        entries.append({**entry, "extensions": ext})
    return {**book, "entries": entries}


def _camel(key: str) -> str:
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
    # Only the portrait is exported: drop references into a CHARX archive; the main icon is the file itself.
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
        # A damaged portrait (a truncated PNG) still exports, on a plain placeholder.
        return embed_card_in_png(to_png(None, character.get("name") or ""), chunks)
