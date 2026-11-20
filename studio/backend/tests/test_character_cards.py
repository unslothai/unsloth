# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import io
import json
import random
import zipfile

import pytest
from PIL import Image

from core.characters import cards, placeholder, png_text

V2 = {"spec": "chara_card_v2", "spec_version": "2.0", "data": {"name": "Mira", "description": "d"}}


def _image(
    fmt: str = "PNG",
    size = (8, 8),
    **save,
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buffer, fmt, **save)
    return buffer.getvalue()


def _with_chunk(
    png: bytes,
    key: bytes,
    body: bytes,
    kind: bytes = b"tEXt",
) -> bytes:
    return png[:-12] + png_text.png_chunk(kind, key + b"\x00" + body) + png[-12:]


def _charx(card_text: str, members: dict | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("card.json", card_text)
        for name, data in (members or {}).items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _entry(**overrides) -> dict:
    entry = {
        "id": 1,
        "keys": ["k"],
        "secondary_keys": ["s"],
        "content": "c",
        "enabled": True,
        "selective": True,
        "case_sensitive": False,
        "insertion_order": 100,
        "position": "before_char",
        "depth": 4,
        "selective_logic": "and_any",
        "extensions": {},
    }
    return {**entry, **overrides}


def _reimported_entry(card: dict) -> dict:
    return cards.parse_card_file(json.dumps(card).encode(), "a.json").fields["character_book"][
        "entries"
    ][0]


@pytest.mark.parametrize("build", [cards.build_card_v2, cards.build_card_v3])
def test_lorebook_logic_and_case_edits_survive_export_and_reimport(build):
    imported = {"selectiveLogic": 0, "case_sensitive": True}
    for ext in ({}, imported):
        entry = _entry(selective_logic = "not_any", case_sensitive = False, extensions = dict(ext))
        character = {"name": "A", "description": "d", "characterBook": {"entries": [entry]}}
        exported = build(character)
        again = _reimported_entry(exported)
        assert again["selective_logic"] == "not_any"
        assert again["case_sensitive"] is False
        assert exported["data"]["character_book"]["entries"][0]["extensions"]["selectiveLogic"] == 2


def test_lorebook_edits_survive_a_png_round_trip():
    entry = _entry(selective_logic = "and_all", case_sensitive = True)
    png = cards.export_png(
        {"name": "A", "description": "d", "characterBook": {"entries": [entry]}}, None
    )
    again = cards.parse_card_file(png, "a.png").fields["character_book"]["entries"][0]
    assert (again["selective_logic"], again["case_sensitive"]) == ("and_all", True)


def test_a_depth_entry_exports_a_spec_position_and_reimports_at_depth():
    entry = _entry(position = "at_depth", depth = 2)
    exported = cards.build_card_v3(
        {"name": "A", "description": "d", "characterBook": {"entries": [entry]}}
    )
    out = exported["data"]["character_book"]["entries"][0]
    assert out["position"] in ("before_char", "after_char")
    assert out["extensions"]["position"] == 4
    assert _reimported_entry(exported)["position"] == "at_depth"


def test_a_sillytavern_world_file_keeps_its_top_level_placement_and_logic():
    world = {
        "entries": {
            "0": {
                "uid": 0,
                "key": ["k"],
                "content": "c",
                "selectiveLogic": 2,
                "position": 4,
                "depth": 1,
            },
            "1": {"uid": 1, "key": ["j"], "content": "c", "position": 1, "disable": True},
        }
    }
    first, second = cards.normalize_book(world)["entries"]
    assert (first["position"], first["selective_logic"], first["depth"]) == (
        "at_depth",
        "not_any",
        1,
    )
    assert (second["position"], second["enabled"]) == ("after_char", False)


def test_a_zero_enabled_flag_disables_the_entry():
    book = cards.normalize_book({"entries": [{"keys": ["k"], "content": "c", "enabled": 0}]})
    assert book["entries"][0]["enabled"] is False


def test_export_turns_a_jpeg_upright_by_its_exif_orientation():
    exif = Image.Exif()
    exif[0x0112] = 6  # rotated 90 degrees, as phones write it
    jpeg = _image("JPEG", (40, 20), exif = exif)
    assert Image.open(io.BytesIO(cards.to_png(jpeg))).size == (20, 40)


def test_export_keeps_a_gif_portraits_transparency():
    image = Image.new("P", (4, 4), 0)
    image.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    image.putpixel((0, 0), 1)
    buffer = io.BytesIO()
    image.save(buffer, "GIF", transparency = 0)
    out = Image.open(io.BytesIO(cards.to_png(buffer.getvalue())))
    assert out.mode == "RGBA"
    assert out.getpixel((2, 2))[3] == 0


@pytest.mark.parametrize(
    "name, hue",
    # Values from features/characters/images.ts nameHue.
    [("Mara Quill", 12), ("Rin Takeda", 250), ("Zoë 🌙", 330), ("", 12)],
)
def test_the_placeholder_uses_the_apps_name_hue(name, hue):
    assert placeholder._name_hue(name) == hue
    top = Image.open(io.BytesIO(cards.to_png(None, name))).convert("RGB").getpixel((200, 0))
    assert max(abs(a - b) for a, b in zip(top, placeholder._hsl_to_rgb(hue, 0.42, 0.62))) <= 2


DEEP = '{"name": "A", "description": "d", "x": ' + "[" * 100_000 + "]" * 100_000 + "}"
HUGE_INT = '{"name": "A", "description": "d", "x": ' + "9" * 5000 + "}"


@pytest.mark.parametrize("text", [DEEP, HUGE_INT], ids = ["deep", "huge-int"])
@pytest.mark.parametrize("wrap", ["json", "charx", "png"])
def test_unreadable_json_is_refused_as_a_card_error(text, wrap):
    data = {
        "json": lambda: text.encode(),
        "charx": lambda: _charx(text),
        "png": lambda: _with_chunk(_image(), b"chara", base64.b64encode(text.encode())),
    }[wrap]()
    with pytest.raises(cards.CardError, match = "nested too deeply"):
        cards.parse_card_file(data, f"a.{wrap}")


def test_a_damaged_charx_is_always_a_card_error():
    v3 = {"spec": "chara_card_v3", "data": {"name": "A", "description": "d"}}
    base = _charx(json.dumps(v3), {"assets/icon/images/main.png": _image()})
    rnd = random.Random(1)
    for _ in range(1500):
        data = bytearray(base)
        for _ in range(rnd.randint(1, 6)):
            data[rnd.randrange(len(data))] = rnd.randrange(256)
        if rnd.random() < 0.2:
            data = data[: rnd.randrange(len(data))]
        try:
            cards.parse_card_file(bytes(data), "a.charx")
        except cards.CardError:
            pass


def test_raw_utf8_json_in_a_text_chunk_is_not_mojibake():
    raw = json.dumps({"name": "Zoë", "description": "café"}, ensure_ascii = False).encode()
    assert (
        cards.parse_card_file(_with_chunk(_image(), b"chara", raw), "a.png").fields["name"] == "Zoë"
    )


@pytest.mark.parametrize("extra", [1, 2, 3])
def test_line_wrapped_base64_without_padding_is_read(extra):
    card = {**V2, "data": {**V2["data"], "description": "d" * extra}}
    encoded = base64.b64encode(json.dumps(card).encode()).rstrip(b"=")
    wrapped = encoded[:10] + b"\n" + encoded[10:]
    parsed = cards.parse_card_file(_with_chunk(_image(), b"chara", wrapped), "a.png")
    assert parsed.fields["description"] == "d" * extra


def test_every_cut_to_fit_is_reported():
    data = {
        "name": "N" * 250,
        "description": "x" * (cards.MAX_FIELD_CHARS + 1),
        "creator": "c" * 250,
        "character_version": "v" * 150,
        "alternate_greetings": [f"g{i}" for i in range(80)],
        "tags": [f"t{i}" for i in range(70)] + ["long" * 40],
    }
    parsed = cards.parse_card_file(json.dumps({**V2, "data": data}).encode(), "a.json")
    fields, warnings = parsed.fields, parsed.warnings
    assert len(fields["name"]) == cards.MAX_SHORT_CHARS
    assert len(fields["character_version"]) == 150
    assert len(fields["alternate_greetings"]) == cards.MAX_GREETINGS
    assert len(fields["tags"]) == cards.MAX_TAGS
    for expected in (
        "name",
        "description",
        "creator",
        "Kept the first 64 of 80 alternate greetings.",
        "Kept the first 64 of 71 tags.",
        "Shortened tags longer than 100 characters.",
    ):
        assert any(expected in warning for warning in warnings), (expected, warnings)
    assert not any("version" in warning for warning in warnings)


def test_a_card_that_fits_has_no_warnings():
    assert cards.parse_card_file(json.dumps(V2).encode(), "a.json").warnings == []


def test_an_imported_portrait_over_the_avatar_limit_is_dropped(monkeypatch):
    png = _with_chunk(_image(size = (64, 64)), b"chara", base64.b64encode(json.dumps(V2).encode()))
    monkeypatch.setattr(cards, "MAX_AVATAR_BYTES", 64)
    parsed = cards.parse_card_file(png, "a.png")
    assert parsed.avatar is None
    assert any("larger than" in warning for warning in parsed.warnings)


def test_a_charx_whose_icon_does_not_decode_says_so():
    v3 = {
        "spec": "chara_card_v3",
        "data": {
            "name": "A",
            "description": "d",
            "assets": [{"type": "icon", "name": "main", "uri": "embeded://i.png"}],
        },
    }
    parsed = cards.parse_card_file(
        _charx(json.dumps(v3), {"i.png": b"\x89PNG\r\n\x1a\nbroken"}), "a.charx"
    )
    assert parsed.avatar is None
    assert any("picture could not be read" in warning for warning in parsed.warnings)


def test_a_json_card_named_charx_is_read_as_json():
    parsed = cards.parse_card_file(json.dumps(V2).encode(), "mislabelled.charx")
    assert parsed.source_format == "json"


def test_a_v1_card_exports_only_v2_data_keys():
    v1 = {
        "name": "A",
        "description": "d",
        "creatorcomment": "note",
        "avatar": "none",
        "chat": "A - 2024",
        "create_date": "x",
        "talkativeness": "0.5",
        "fav": False,
    }
    parsed = cards.parse_card_file(json.dumps(v1).encode(), "a.json")
    character = {**cards.character_fields(parsed), "card": parsed.card}
    data = cards.build_card_v2(character)["data"]
    assert not {"creatorcomment", "avatar", "chat", "create_date", "talkativeness", "fav"} & set(
        data
    )
    assert data["creator_notes"] == "note"


def test_the_export_placeholder_carries_the_apps_initials():
    from PIL import Image

    from core.characters.placeholder import _initials, placeholder_png

    assert _initials("Inspector Theo Lindqvist") == "TL"
    assert _initials("Captain Ivo") == "I"
    assert _initials("Dr.") == "D"
    assert _initials("   ") == "?"
    plain = Image.open(io.BytesIO(placeholder_png("山田"))).convert("RGB")
    lettered = Image.open(io.BytesIO(placeholder_png("Rin Takeda"))).convert("RGB")
    # White initials over the middle of the gradient; no boxes for scripts the font lacks.
    assert (255, 255, 255) in {lettered.getpixel((x, 300)) for x in range(120, 280)}
    assert (255, 255, 255) not in {plain.getpixel((x, 300)) for x in range(0, 400, 4)}
