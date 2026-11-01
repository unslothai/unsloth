# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The bundled example flows, built in code so the editor's Examples shelf and the tests read one source.
`python -m core.flows.examples.build_examples` prints each example's id and node count."""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path
from typing import Any, Optional

from core.flows.recipe_import import flow_from_recipe
from core.flows.schema import FlowDocument

CHAT = {"repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF", "variant": "Q4_K_M"}
VISION = {"repo": "unsloth/Qwen3.5-4B-GGUF", "variant": "Q4_K_M"}
IMAGE = {"repo": "black-forest-labs/FLUX.2-klein-4B", "variant": None}
VIDEO = {"repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "variant": None}
TTS = {"repo": "audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF", "variant": None}

CLONE = {"repo": "audio-cpp/audio.cpp-gguf/Qwen3-TTS-12Hz-0.6B-Base-GGUF", "variant": None}
MUSIC = {"repo": "audio-cpp/audio.cpp-gguf/ACE-Step1.5-GGUF", "variant": None}
RIVAL = {"repo": "unsloth/gemma-3-4b-it-GGUF", "variant": "Q4_K_M"}
COLAB_NB = "https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/"
KREA = {"repo": "krea/Krea-2-Turbo", "variant": None}
ZIMAGE = {"repo": "Tongyi-MAI/Z-Image-Turbo", "variant": None}
FLUX1 = {"repo": "black-forest-labs/FLUX.1-dev", "variant": None}
RESUME_FIT = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "requirement": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["requirement", "evidence"],
            },
        },
        "gaps": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
        "keywords": {"type": "array", "maxItems": 10, "items": {"type": "string"}},
    },
    "required": ["matches", "gaps", "keywords"],
}
LOCALIZATION_STRINGS = {
    "welcome": "Welcome back, {name}!",
    "cart_count": "You have {count} items in your cart.",
    "checkout": "Checkout",
    "free_shipping": "Free shipping on orders over {amount}.",
    "error_payment": "Your payment did not go through. Please try another card.",
}
INVENTORY_STOCK = [
    {"sku": "MUG-01", "item": "Sloth mug", "on_hand": 24, "unit_cost": 6.5},
    {"sku": "TEE-M", "item": "T-shirt (M)", "on_hand": 40, "unit_cost": 9.0},
    {"sku": "TEE-L", "item": "T-shirt (L)", "on_hand": 35, "unit_cost": 9.0},
    {"sku": "CAP-01", "item": "Cap", "on_hand": 18, "unit_cost": 7.25},
    {"sku": "STK-10", "item": "Sticker pack", "on_hand": 120, "unit_cost": 1.2},
    {"sku": "BAG-01", "item": "Tote bag", "on_hand": 12, "unit_cost": 4.0},
]
INVENTORY_COUNT = [
    {"sku": "MUG-01", "counted": 24},
    {"sku": "TEE-M", "counted": 37},
    {"sku": "TEE-L", "counted": 35},
    {"sku": "CAP-01", "counted": 20},
    {"sku": "STK-10", "counted": 96},
    {"sku": "PIN-05", "counted": 15},
]
REQ = {
    "chat": {
        "ref": "writer",
        "kind": "chat",
        "note": "Any instruct chat model works; 4B is fast and follows JSON schemas well.",
        **CHAT,
    },
    "vision": {"ref": "judge", "kind": "vision", "note": "Any vision chat model.", **VISION},
    "image": {
        "ref": "painter",
        "kind": "image",
        "note": "Fast 4B image model with reference/edit support; any image model works.",
        **IMAGE,
    },
    "video": {"ref": "director", "kind": "video", "note": "5B text+image to video.", **VIDEO},
    "embedding": {
        "ref": "embedder",
        "kind": "embedding",
        "note": "Studio's local embedding model for knowledge bases (Settings can change it); runs offline once "
        "downloaded.",
        "repo": "unsloth/bge-small-en-v1.5",
        "variant": None,
    },
    "tts": {
        "ref": "voice",
        "kind": "tts",
        "note": "82M multi-voice TTS (voices af_heart, am_michael, ...).",
        **TTS,
    },
    "clone": {
        "ref": "dubber",
        "kind": "tts",
        "note": "0.6B voice-cloning TTS that speaks 10 languages; any cloning voice model works.",
        **CLONE,
    },
    "music": {
        "ref": "band",
        "kind": "audio",
        "note": "Songs with vocals from lyrics; any music model works.",
        **MUSIC,
    },
}


class B:
    """Tiny builder: nodes on a grid (col, row), edges by `a.port -> b.port`."""

    def __init__(
        self,
        id: str,
        name: str,
        kind: str,
        description: str,
        *,
        tags: list[str],
        seed: int = 7,
    ):
        self.doc: dict[str, Any] = {
            "format": "unsloth-flow",
            "version": 1,
            "id": id,
            "name": name,
            "description": description,
            "kind": kind,
            "seed": seed,
            "params": [],
            "requirements": {"models": []},
            "nodes": [],
            "edges": [],
            "groups": [],
            "notes": [],
            "assets": [],
            "baked": {},
            "editor": {"viewport": {"x": 40, "y": 40, "zoom": 0.8}},
            "meta": {"tags": tags, "example": True, "author": "Unsloth"},
        }
        self._e = 0

    def need(
        self,
        *kinds: str,
        optional: tuple[str, ...] = (),
    ) -> "B":
        for k in kinds + optional:
            self.doc["requirements"]["models"].append({**REQ[k], "optional": k in optional})
        return self

    def n(
        self,
        node_id: str,
        node_type: str,
        col: float,
        row: float,
        /,
        *,
        label: Optional[str] = None,
        group: Optional[str] = None,
        **params: Any,
    ) -> str:
        node = {
            "id": node_id,
            "type": node_type,
            "v": 1,
            "params": params,
            "pos": [round(col * 320), round(row * 190)],
        }
        if label:
            node["label"] = label
        if group:
            node["group"] = group
        self.doc["nodes"].append(node)
        return node_id

    def e(
        self,
        src: str,
        dst: str,
        kind: str = "data",
    ) -> None:
        s, sp = src.split(".")
        t, tp = dst.split(".") if kind == "data" else (dst, "in")
        self._e += 1
        self.doc["edges"].append(
            {
                "id": f"e{self._e}",
                "source": s,
                "source_port": sp,
                "target": t,
                "target_port": tp,
                "kind": kind,
            }
        )

    def f(self, src: str, dst: str) -> None:
        self.e(src, dst, "flow")

    def group(
        self, id: str, label: str, rect: tuple[float, float, float, float], color: str
    ) -> None:
        self.doc["groups"].append(
            {
                "id": id,
                "label": label,
                "color": color,
                "rect": [
                    round(rect[0] * 320) - 30,
                    round(rect[1] * 190) - 50,
                    round(rect[2] * 320),
                    round(rect[3] * 190),
                ],
            }
        )

    def note(
        self,
        id: str,
        text: str,
        col: float,
        row: float,
        w: float = 1.6,
        h: float = 1.0,
        color: str = "#fde68a",
    ) -> None:
        self.doc["notes"].append(
            {
                "id": id,
                "text": text,
                "color": color,
                "rect": [round(col * 320), round(row * 190), round(w * 320), round(h * 190)],
            }
        )

    def param(self, id: str, type: str, default: Any, label: str) -> None:
        self.doc["params"].append({"id": id, "type": type, "default": default, "label": label})


def sloth_moon() -> B:
    b = B(
        "sloth-moon",
        "The Sloth Who Lost the Moon",
        "story",
        "A cozy branching visual novel. Mo the sloth wakes up and the moon is gone. Three endings, a secret one if "
        "you are kind. Press Play to read it right away (painted-on-the-fly backgrounds), or Run first to paint "
        "every background and portrait with your image model.",
        tags = ["story", "visual novel", "beginner"],
        seed = 1111,
    ).need("image")
    style = "soft storybook watercolor, warm lantern light, gentle pastel night palette"
    b.n(
        "start",
        "story.start",
        0,
        2,
        title = "The Sloth Who Lost the Moon",
        vars = {"kindness": 0, "courage": 0},
        style = style,
        mode = "fixed",
    )
    chars = [
        (
            "mo",
            "Mo",
            "a small sleepy three-toed sloth in a knitted blue scarf, big kind eyes",
            "#7dd3fc",
            "left",
        ),
        (
            "pip",
            "Pip",
            "a tiny glowing firefly with round spectacles and a lantern-shaped tail",
            "#fde047",
            "right",
        ),
        (
            "otto",
            "Ottoline",
            "a wise old barn owl wearing a patchwork shawl and a pocket watch",
            "#c4b5fd",
            "right",
        ),
    ]
    for i, (cid, name, look, color, side) in enumerate(chars):
        b.n(
            f"art_{cid}",
            "image.generate",
            0,
            4 + i,
            label = f"{name} portrait",
            model = IMAGE,
            prompt = f"character portrait of {name}, {look}, upper body, plain pale background, {style}",
            size = "768x1024",
            seed = 200 + i,
        )
        b.n(f"cut_{cid}", "image.remove_bg", 1, 4 + i, subject = f"the character {name}")
        b.n(cid, "story.character", 2, 4 + i, name = name, look = look, color = color, side = side)
        b.e(f"art_{cid}.image", f"cut_{cid}.image")
        b.e(f"cut_{cid}.image", f"{cid}.portrait")
    b.n(
        "faces_mo",
        "story.expressions",
        1,
        3.2,
        label = "Mo's expressions",
        model = IMAGE,
        name = "Mo",
        look = chars[0][2],
        style = style,
        expressions = ["worried", "happy"],
        seed = 200,
    )
    b.e("art_mo.image", "faces_mo.portrait")
    b.e("faces_mo.expressions", "mo.expressions")
    scenes = [
        (
            "porch",
            1,
            0,
            "The Porch",
            "a wooden treehouse porch high in a jungle tree at night, empty dark sky, hanging lanterns",
            "Narrator: Every night, Mo the sloth fell asleep watching the moon.\n"
            "Narrator: But tonight, the sky was empty.\nMo (worried): The moon... it's gone!\n"
            "Pip: Psst! Down here! I saw something shiny roll into the Whispering Woods.\n"
            "Mo: Rolled? The moon doesn't roll. ...Does it?",
        ),
        (
            "woods",
            3,
            -1,
            "The Whispering Woods",
            "a misty glowing forest path with giant ferns and blue mushrooms at night",
            "Narrator: The woods hummed with tiny voices.\nPip: Stay close. The trees here gossip.\n"
            "Narrator: Something silver glinted under a pile of leaves.",
        ),
        (
            "oak",
            3,
            1,
            "The Old Oak",
            "a huge ancient oak tree with a round door and a lit window, starry clearing",
            "Ottoline: Hoo. A sloth awake at night? This must be serious.\n"
            "Mo: The moon is missing. Have you seen it?\n"
            "Ottoline: The moon never goes missing, little one. It only hides when it's sad.",
        ),
        (
            "fog",
            5,
            -2,
            "Fog Hollow",
            "a hollow filled with thick silver fog, faint shapes of trees, mysterious",
            "[scene with fade]\nNarrator: Fog swallowed the path.\nPip: I... can't see my own glow.\nMo: Then I'll carry you. Slowly. Very slowly.",
        ),
        (
            "dance",
            5,
            -1,
            "Firefly Glade",
            "a glade full of hundreds of dancing fireflies, magical golden light",
            "Narrator: A thousand fireflies rose like sparks.\nPip: My cousins! They're throwing a party!\n"
            "Pip: They say the moon came by earlier... looking lonely.",
        ),
        (
            "lake",
            5,
            1,
            "Mirror Lake",
            "a perfectly still lake reflecting the stars, reeds, a small wooden dock",
            "Narrator: At the lake, the water was so still it looked like a second sky.\n"
            "Mo (happy): Look! The moon is in the water!\nOttoline: Or is it?",
        ),
    ]
    for sid, col, row, title, bg, lines in scenes:
        b.n(
            f"bg_{sid}",
            "image.generate",
            col,
            row + 2 - 0.5,
            label = f"{title} background",
            model = IMAGE,
            prompt = f"{bg}, {style}, wide establishing shot, no people, no text",
            size = "1344x768",
            seed = 100 + col * 10 + row,
        )
        b.n(sid, "story.scene", col + 1, row + 2, title = title, background_prompt = bg, lines = lines)
        b.e(f"bg_{sid}.image", f"{sid}.background")
    for cid, sid in (("mo", "porch"), ("pip", "porch"), ("otto", "oak")):
        b.e(f"{cid}.character", f"{sid}.characters")
    b.n(
        "c_porch",
        "story.choice",
        2.5,
        2.6,
        prompt = "Where should Mo look first?",
        choices = [
            "Follow Pip into the Whispering Woods",
            "Climb down to ask Ottoline at the Old Oak",
        ],
        effects = {"0": {"courage": {"op": "+", "args": [{"var": "courage"}, 1]}}},
    )
    b.n("rnd_woods", "story.random", 4.5, 0.6, weights = [1, 2])
    b.n(
        "c_dance",
        "story.choice",
        6.5,
        0.6,
        prompt = "Pip's cousins look so happy.",
        choices = ["Dance with the fireflies", "Ask them to help find the moon"],
        effects = {
            "0": {"kindness": {"op": "+", "args": [{"var": "kindness"}, 1]}},
            "1": {"courage": {"op": "+", "args": [{"var": "courage"}, 1]}},
        },
    )
    b.n(
        "c_fog",
        "story.choice",
        6.5,
        -0.4,
        prompt = "Mo hears a soft sob in the fog.",
        choices = ["Follow the sound", "Hum a lullaby to calm everyone"],
        effects = {"1": {"kindness": {"op": "+", "args": [{"var": "kindness"}, 2]}}},
    )
    b.n(
        "c_oak",
        "story.choice",
        4.5,
        3.6,
        prompt = "Ottoline offers you tea.",
        choices = ["Drink the tea and listen", "Hurry to the lake"],
        effects = {"0": {"kindness": {"op": "+", "args": [{"var": "kindness"}, 1]}}},
    )
    b.n(
        "br_lake",
        "story.branch",
        7,
        3,
        label = "Kind enough?",
        condition = {"op": ">=", "args": [{"var": "kindness"}, 2]},
    )
    b.n(
        "end_good",
        "story.ending",
        8,
        1,
        title = "The Moon Comes Home",
        kind = "good",
        text = "Narrator: The moon, who had only felt forgotten, rose back into the sky.\n"
        "Mo (happy): Goodnight, moon. We'll watch you every night.",
    )
    b.n(
        "end_secret",
        "story.ending",
        8,
        2.2,
        title = "Starlit Friends",
        kind = "secret",
        text = "Narrator: The moon floated down from its hiding place in the lake and hugged everyone, very carefully.\n"
        "Ottoline: Hoo. Kindness, it seems, is brighter than any lantern.\n"
        "Narrator: From that night on, the moon visited the porch every Sunday for tea.",
    )
    b.n(
        "end_nap",
        "story.ending",
        8,
        3.6,
        title = "A Very Long Nap",
        kind = "neutral",
        text = "Narrator: Mo sat on the dock to think... and fell asleep for three days.\n"
        "Narrator: When Mo woke up, the moon was back. Nobody ever found out where it went.",
    )
    b.f("start.next", "porch")
    b.f("porch.next", "c_porch")
    b.f("c_porch.choice_1", "woods")
    b.f("c_porch.choice_2", "oak")
    b.f("woods.next", "rnd_woods")
    b.f("rnd_woods.path_1", "fog")
    b.f("rnd_woods.path_2", "dance")
    b.f("fog.next", "c_fog")
    b.f("c_fog.choice_1", "lake")
    b.f("c_fog.choice_2", "lake")
    b.f("dance.next", "c_dance")
    b.f("c_dance.choice_1", "lake")
    b.f("c_dance.choice_2", "end_good")
    b.f("oak.next", "c_oak")
    b.f("c_oak.choice_1", "lake")
    b.f("c_oak.choice_2", "end_nap")
    b.f("lake.next", "br_lake")
    b.f("br_lake.then", "end_secret")
    b.f("br_lake.else", "end_good")
    b.n(
        "art",
        "story.scene_art",
        0,
        0,
        label = "Paint missing backgrounds (live)",
        model = IMAGE,
        style = style,
    )
    b.note(
        "n1",
        "How it works\n\nStory -> Scene -> Choice nodes are the script (white story edges). Image nodes feed each "
        "Scene's background and each Character's portrait.\n\nPlay right away: scenes without a painting are "
        "painted live by the Scene Art node (or drawn as soft gradients with no model).\nRun first to paint "
        "everything ahead of time, then Share as a baked .unslothflow so friends can play with no models.",
        -1.8,
        0,
        1.6,
        1.6,
    )
    return b


def lantern_picnic() -> B:
    b = B(
        "lantern-picnic",
        "Lantern Picnic (no models needed)",
        "story",
        "A tiny branching story that plays instantly on any computer: no models, no downloads. Pack a picnic, "
        "pick a spot, and see which of three endings you find. A good first flow to remix.",
        tags = ["story", "no models", "beginner"],
        seed = 42,
    )
    b.n(
        "start",
        "story.start",
        0,
        1,
        title = "Lantern Picnic",
        vars = {"snacks": 0, "friends": 0},
        style = "cozy paper-lantern evening",
        mode = "fixed",
    )
    b.n(
        "hana",
        "story.character",
        0,
        3,
        name = "Hana",
        look = "a cheerful red panda with a picnic basket",
        color = "#fb923c",
        side = "left",
    )
    b.n(
        "bun",
        "story.character",
        0,
        4,
        name = "Bun",
        look = "a shy grey bunny with a paper lantern",
        color = "#a5b4fc",
        side = "right",
    )
    b.n(
        "s_start",
        "story.scene",
        1,
        1,
        title = "Golden Hour",
        lines = "Narrator: The sun is melting into the hills. Perfect picnic weather.\n"
        "Hana: I packed rice balls, peaches and exactly one enormous cake.\n"
        "Bun: Where should we go?",
    )
    b.e("hana.character", "s_start.characters")
    b.e("bun.character", "s_start.characters")
    b.n(
        "c1",
        "story.choice",
        2,
        1,
        prompt = "Pick a picnic spot.",
        choices = [
            "The hill with the old windmill",
            "The creek with the stepping stones",
            "The lantern market in town",
        ],
        effects = {
            "0": {"snacks": 1},
            "1": {"snacks": 2},
            "2": {"friends": {"op": "+", "args": [{"var": "friends"}, 2]}},
        },
    )
    b.n(
        "s_hill",
        "story.scene",
        3,
        0,
        title = "Windmill Hill",
        lines = "Narrator: The windmill creaks a slow lullaby.\nBun: The wind keeps stealing my napkin!\n"
        "Hana: That's not the wind. That's a goat.",
    )
    b.n(
        "s_creek",
        "story.scene",
        3,
        1,
        title = "Stepping Stones",
        lines = "Narrator: Water chatters over smooth stones.\nHana: One... two... whoa--\n"
        "Narrator: Splash. The cake survives. Hana's dignity does not.",
    )
    b.n(
        "s_market",
        "story.scene",
        3,
        2,
        title = "Lantern Market",
        lines = "Narrator: A hundred paper lanterns sway over the market.\nBun: Everyone is sharing food here!\n"
        "Hana: Then we share the cake. Obviously.",
    )
    b.n("rnd", "story.random", 4, 0.5, label = "Surprise guest?", weights = [1, 1])
    b.n(
        "s_goat",
        "story.scene",
        5,
        0,
        title = "An Uninvited Goat",
        lines = "Narrator: The goat sits down on your blanket like it was invited.\nBun: ...Should we name it?\n"
        "Hana: Clementine.",
    )
    b.n(
        "set_goat", "story.set", 6, 0, set = {"friends": {"op": "+", "args": [{"var": "friends"}, 2]}}
    )
    b.n(
        "s_stars",
        "story.scene",
        5,
        1,
        title = "First Stars",
        lines = "Narrator: The first stars blink awake.\nBun: Make a wish!\nHana: I wish for more cake.",
    )
    b.n(
        "br",
        "story.branch",
        7,
        1,
        label = "Made friends?",
        condition = {"op": ">=", "args": [{"var": "friends"}, 2]},
    )
    b.n(
        "end_party",
        "story.ending",
        8,
        0.5,
        title = "The Biggest Picnic",
        kind = "good",
        text = "Narrator: By midnight, half the town has joined your picnic. Nobody remembers who brought the goat.",
    )
    b.n(
        "end_quiet",
        "story.ending",
        8,
        1.5,
        title = "Just the Two of Us",
        kind = "neutral",
        text = "Narrator: Two friends, one blanket, a sky full of stars. Some picnics are perfect exactly as they are.",
    )
    b.n(
        "end_cake",
        "story.ending",
        8,
        2.5,
        title = "The Cake Incident",
        kind = "secret",
        text = "Hana: Wait. Where is the cake?\nBun: ...\nNarrator: Bun has frosting on its nose. The case is closed.",
    )
    b.n(
        "br_cake",
        "story.branch",
        7,
        2.2,
        label = "Lots of snacks?",
        condition = {"op": ">=", "args": [{"var": "snacks"}, 2]},
    )
    b.f("start.next", "s_start")
    b.f("s_start.next", "c1")
    b.f("c1.choice_1", "s_hill")
    b.f("c1.choice_2", "s_creek")
    b.f("c1.choice_3", "s_market")
    b.f("s_hill.next", "rnd")
    b.f("rnd.path_1", "s_goat")
    b.f("rnd.path_2", "s_stars")
    b.f("s_goat.next", "set_goat")
    b.f("set_goat.next", "s_stars")
    b.f("s_stars.next", "br")
    b.f("br.then", "end_party")
    b.f("br.else", "end_quiet")
    b.f("s_creek.next", "br_cake")
    b.f("br_cake.then", "end_cake")
    b.f("br_cake.else", "s_stars")
    b.f("s_market.next", "br")
    b.note(
        "n1",
        "No models needed\n\nEvery scene is written by hand and the player paints soft procedural backdrops. "
        "Try editing a line, adding a Choice, or connecting an Image node to a Scene's background.",
        -1.8,
        0,
        1.6,
        1.2,
    )
    return b


def infinite_dungeon() -> B:
    b = B(
        "infinite-dungeon",
        "Infinite Dungeon Tales",
        "story",
        "A dungeon crawl that never ends the same way twice. You write the opening; the LLM writes every room after "
        "it from your choices, health and gold, and the image model paints each room as you enter.",
        tags = ["story", "infinite", "rpg", "random"],
        seed = 777,
    ).need("chat", "image")
    style = "dark fantasy illustration, torchlight, rich shadows, painterly"
    b.n(
        "start",
        "story.start",
        0,
        1,
        title = "Infinite Dungeon Tales",
        vars = {"hp": 10, "gold": 0, "depth": 0},
        style = style,
        mode = "infinite",
    )
    b.n(
        "hero",
        "story.character",
        0,
        2.4,
        name = "Rook",
        look = "a scrappy young adventurer with a dented helmet",
        color = "#fbbf24",
        side = "left",
    )
    b.n(
        "door",
        "story.scene",
        1,
        1,
        title = "The Mouth of the Dungeon",
        generate_background = True,
        background_prompt = "a crumbling stone dungeon entrance carved like a giant open mouth, moss, a single torch",
        lines = "Narrator: Legends say the dungeon rearranges itself every night.\n"
        "Rook: Ten health, zero gold, and a very good feeling about this.\n"
        "Narrator: Cold air breathes out of the dark.",
    )
    b.e("hero.character", "door.characters")
    b.n(
        "c",
        "story.choice",
        2,
        1,
        prompt = "How do you enter?",
        choices = [
            "Light a torch and walk straight in",
            "Search the entrance for traps",
            "Sing loudly to announce yourself",
        ],
        effects = {"0": {"depth": 1}, "1": {"gold": 2, "depth": 1}, "2": {"hp": 9, "depth": 1}},
    )
    b.f("start.next", "door")
    b.f("door.next", "c")
    b.n(
        "next",
        "story.next_scene",
        3,
        0,
        label = "Dungeon Master (live)",
        model = CHAT,
        choices = 3,
        instructions = "This is a dungeon crawler. Track `hp` and `gold` with effects (damage is negative hp, "
        "treasure is positive gold; also add 1 to `depth` on every choice). If hp would reach 0, write a "
        "bad ending. Around depth 10, offer a way out with a good ending. Keep each scene short and vivid.",
    )
    b.n(
        "art",
        "story.scene_art",
        3,
        2,
        label = "Room Painter (live)",
        model = IMAGE,
        size = "1344x768",
        style = style,
    )
    b.note(
        "n1",
        "Infinite mode\n\nThe Choice's exits are left unconnected on purpose: when you pick one, the player asks "
        "the Dungeon Master node to write the next room, and the Room Painter paints it. Change the "
        "instructions to make it a space station, a haunted mall, or a cooking show.",
        4,
        0,
        1.6,
        1.4,
    )
    return b


def pocket_village() -> B:
    b = B(
        "pocket-village",
        "Pocket Village",
        "world",
        "Walk around a tiny generated village. The LLM invents the villagers, their quests and items; the image "
        "model draws every sprite and tile; villagers talk back in character when you chat with them.",
        tags = ["world", "game", "rpg", "npc"],
        seed = 2024,
    ).need("chat", "image")
    b.param(
        "premise",
        "text",
        "A cozy mushroom village where the annual Lantern Festival starts tonight, "
        "but the festival's golden bell has gone missing.",
        "Premise",
    )
    b.n(
        "premise",
        "input.param",
        0,
        1,
        name = "premise",
        type = "text",
        default = "A cozy mushroom village where the annual Lantern Festival starts tonight, but the festival's golden "
        "bell has gone missing.",
    )
    b.n(
        "gen",
        "world.generator",
        1,
        1,
        model = CHAT,
        npcs = 4,
        items = 4,
        biome = "meadow",
        style = "cute 16-bit pixel art, top-down RPG, soft colours",
    )
    b.n(
        "fern",
        "world.npc",
        1,
        2.4,
        name = "Grandma Fern",
        persona = "The village elder, a tiny old mushroom lady who knits scarves for everyone and knows every secret. "
        "Speaks in warm proverbs. Secretly hid the bell to test the young ones.",
        greeting = "Ah, a new face! Sit, sit. Have you eaten?",
    )
    b.n("art", "world.art", 2, 1, model = IMAGE, tiles = True, sprites = True)
    b.n("play", "world.play", 3, 1)
    b.n(
        "reply",
        "world.npc_reply",
        3,
        2.4,
        label = "Villager voices (live)",
        model = CHAT,
        max_tokens = 160,
        rules = "Stay in character. Keep replies under 50 words. Drop gentle hints about quests.",
    )
    b.e("premise.value", "gen.premise")
    b.e("gen.world", "art.world")
    b.e("art.world", "play.world")
    b.note(
        "n1",
        "Run, then Play\n\nRun paints the sprites and tiles (about one image per villager, item and tile). "
        "Play: WASD or arrows (or click) to walk, E or Space to talk and pick things up, I for inventory. "
        "Grandma Fern is hand-written: edit her persona, or add more NPC nodes.",
        0,
        3.4,
        1.8,
        1.2,
    )
    return b


def lantern_isle_3d() -> B:
    b = B(
        "lantern-isle-3d",
        "Lantern Isle 3D",
        "world",
        "Pocket Village in first person: the same generated villagers, quests and art, walked through in 3D. "
        "Mouse to look, WASD to move, E to talk. The World Player node switches between 3D and the map.",
        tags = ["world", "game", "rpg", "npc", "3d"],
        seed = 2031,
    ).need("chat", "image")
    premise = (
        "A small island of lantern makers at dusk. The lighthouse lamp went dark last night and the boats are due "
        "home before the tide turns."
    )
    b.param("premise", "text", premise, "Premise")
    b.n("premise", "input.param", 0, 1, name = "premise", type = "text", default = premise)
    b.n(
        "gen",
        "world.generator",
        1,
        1,
        model = CHAT,
        npcs = 4,
        items = 4,
        biome = "beach",
        style = "cute storybook illustration, soft lantern light, full body, plain background",
    )
    b.n(
        "keeper",
        "world.npc",
        1,
        2.4,
        name = "Keeper Ines",
        persona = "The old lighthouse keeper, gruff but kind, who has never once let the lamp go out until now. "
        "Talks about the sea like an old friend. Suspects the gulls stole the brass wick key.",
        greeting = "Mind the rocks. The light is out and I am in no mood for shipwrecks.",
    )
    b.n("art", "world.art", 2, 1, model = IMAGE, tiles = True, sprites = True)
    b.n("play", "world.play", 3, 1, view = "3d", time_of_day = "dusk")
    b.n(
        "reply",
        "world.npc_reply",
        3,
        2.4,
        label = "Islander voices (live)",
        model = CHAT,
        max_tokens = 160,
        rules = "Stay in character. Keep replies under 50 words. Drop gentle hints about quests.",
    )
    b.e("premise.value", "gen.premise")
    b.e("gen.world", "art.world")
    b.e("art.world", "play.world")
    b.note(
        "n1",
        "Run, then Play\n\nRun paints the villagers and items. Play opens in 3D: click the view to look around "
        "with the mouse, WASD or arrows to walk, E to talk and pick things up, Esc to free the mouse. Set the World "
        "Player node's View to 2d for the map, or change its Time of day.",
        0,
        3.4,
        1.8,
        1.2,
    )
    return b


def mayor_meadow() -> B:
    b = B(
        "mayor-meadow",
        "Mayor of Meadowbrook",
        "world",
        "Mayor mode: zone homes, shops and factories, lay roads and power lines, and watch a Micropolis-style "
        "simulation grow the town. Villagers move into the houses; the chat model is the city advisor and speaks "
        "for the residents once a month, but never places a tile.",
        tags = ["world", "game", "city", "simulation"],
        seed = 1989,
    ).need("chat")
    world = {
        "title": "Meadowbrook",
        "seed": 1989,
        "biome": "meadow",
        "width": 48,
        "height": 36,
        "intro_text": "Meadowbrook is a meadow with a handful of settlers and a mayor with 20,000 dollars. Build a "
        "power plant, zone a block, and connect it all with roads and power lines.",
        "player": {"name": "Mayor"},
        "npcs": [
            {
                "name": "Rosa Park",
                "persona": "A retired teacher who wants quiet streets and clean air.",
                "greeting": "Welcome, Mayor. Mind you keep the factories away from my roses.",
            },
            {
                "name": "Dev Patel",
                "persona": "A young shopkeeper who wants busy roads past his door.",
                "greeting": "More people means more customers. Build, build, build!",
            },
            {
                "name": "Old Bram",
                "persona": "A grumpy miller who distrusts taxes and loves the old meadow.",
                "greeting": "Hmph. Another mayor with big plans.",
            },
            {
                "name": "Lena Fox",
                "persona": "A factory foreman who wants steady power and jobs.",
                "greeting": "Power and jobs, Mayor. That is all we ask.",
            },
        ],
        "city": {"funds": 20000, "tax": 7},
    }
    b.n("world", "world.spec", 0, 1, world = world)
    b.n("play", "world.play", 1, 1, view = "2d")
    b.e("world.world", "play.world")
    b.n(
        "reply",
        "world.npc_reply",
        2,
        1,
        label = "Advisor and residents (live)",
        model = CHAT,
        max_tokens = 160,
        rules = "Stay in character. Keep replies under 50 words. Talk about life in the growing town.",
    )
    b.note(
        "about",
        "Mayor mode after Micropolis, the GPL release of the original SimCity: residential, commercial and industrial "
        "demand, a power flood fill, traffic as a random-walk trip test, and land value, pollution and crime overlays, "
        "all deterministic in the player. Pick a tool in the Mayor panel and click the map (zones are 3x3). One game "
        "month is 24 seconds; the advisor and up to two resident petitions cost one model call a month.",
        0,
        2.2,
        1.8,
        1.4,
    )
    return b


def comic_strip() -> B:
    b = B(
        "comic-strip",
        "Comic Strip Machine",
        "pipeline",
        "One sentence in, a four-panel comic out. The LLM writes the script, the image model draws a character "
        "reference, then each panel with that reference so the hero looks the same every time.",
        tags = ["image", "comic", "for each"],
        seed = 314,
    ).need("chat", "image")
    b.n(
        "idea",
        "input.text",
        0,
        1,
        value = "A sloth enters a speed-eating contest and wins in the most unexpected way.",
    )
    b.n(
        "script",
        "llm.generate",
        1,
        1,
        label = "Comic writer",
        model = CHAT,
        temperature = 0.9,
        max_tokens = 1200,
        system = "You write funny four-panel comics. Panel descriptions are visual and self-contained.",
        prompt = "Write a four-panel comic about: {{prompt}}",
        json_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "hero_look": {"type": "string"},
                "panels": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "string"},
                },
                "captions": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "string"},
                },
            },
            "required": ["title", "hero_look", "panels", "captions"],
        },
    )
    b.e("idea.text", "script.prompt")
    b.n("look", "json.get", 2, 0, path = "hero_look")
    b.n("panels", "json.get", 2, 1, path = "panels")
    b.n("captions", "json.get", 2, 2, path = "captions")
    for x in ("look", "panels", "captions"):
        b.e("script.json", f"{x}.json")
    b.n(
        "ref",
        "image.generate",
        3,
        0,
        label = "Hero reference",
        model = IMAGE,
        size = "768x1024",
        style = "comic book character sheet, full body, plain white background, bold ink lines, flat colors",
        seed = 5,
    )
    b.e("look.value", "ref.prompt")
    b.n("each", "logic.for_each", 3.2, 1.6, group = "panels_loop", max_items = 4)
    b.n(
        "draw",
        "image.generate",
        4.2,
        1.6,
        group = "panels_loop",
        label = "Draw panel",
        model = IMAGE,
        size = "1024x1024",
        style = "comic book panel, bold ink lines, flat bright colors, expressive, no text, no lettering, no speech bubbles",
        seed = 9,
    )
    b.group("panels_loop", "For each panel", (3.2, 1.2, 2.2, 1.6), "#ec4899")
    b.e("panels.value", "each.items")
    b.e("each.item", "draw.prompt")
    b.e("ref.image", "draw.reference")
    b.e("draw.image", "each.collect")
    b.n(
        "page",
        "image.grid",
        5.6,
        1,
        columns = 2,
        gap = 18,
        background = "#fffaf0",
        caption_style = "comic",
        cell_size = 768,
    )
    b.e("each.results", "page.images")
    b.e("captions.value", "page.captions")
    b.n("show", "output.preview", 6.6, 1)
    b.e("page.image", "show.value")
    b.n("save", "output.library", 6.6, 2, folder = "Comics")
    b.e("page.image", "save.value")
    return b


def podcast() -> B:
    b = B(
        "podcast-in-a-box",
        "Podcast in a Box",
        "pipeline",
        "Type a topic, get a two-host podcast episode: the LLM writes a lively script, each line is spoken by one "
        "of two voices, the clips are joined, and the image model paints cover art.",
        tags = ["audio", "tts", "podcast", "switch"],
        seed = 88,
    ).need("chat", "tts", optional = ("image",))
    b.n("topic", "input.text", 0, 1, value = "Why do cats knock things off tables?")
    b.n(
        "script",
        "llm.generate",
        1,
        1,
        label = "Script writer",
        model = CHAT,
        temperature = 0.9,
        max_tokens = 2000,
        system = "You write a short, funny podcast between host A (curious, excitable) and host B (dry expert). "
        "12 lines, alternating, each under 30 words. No stage directions.",
        prompt = "Episode topic: {{prompt}}",
        json_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "lines": {
                    "type": "array",
                    "minItems": 6,
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "properties": {
                            "host": {"type": "string", "enum": ["A", "B"]},
                            "text": {"type": "string"},
                        },
                        "required": ["host", "text"],
                    },
                },
            },
            "required": ["title", "lines"],
        },
    )
    b.e("topic.text", "script.prompt")
    b.n("lines", "json.get", 2, 1, path = "lines")
    b.e("script.json", "lines.json")
    b.n("each", "logic.for_each", 3, 1, group = "lines_loop", max_items = 16)
    b.n("text", "json.get", 4, 0.6, group = "lines_loop", path = "text")
    b.n("host", "json.get", 4, 1.6, group = "lines_loop", path = "host")
    b.n("route", "logic.switch", 5, 1, group = "lines_loop", cases = ["A", "B", "-"])
    b.n(
        "voice_a",
        "audio.speak",
        6,
        0.5,
        group = "lines_loop",
        label = "Host A",
        model = TTS,
        voice = "af_heart",
    )
    b.n(
        "voice_b",
        "audio.speak",
        6,
        1.5,
        group = "lines_loop",
        label = "Host B",
        model = TTS,
        voice = "am_michael",
    )
    b.n("either", "logic.merge", 7, 1, group = "lines_loop")
    b.group("lines_loop", "For each line", (3, 0.2, 5, 2.4), "#10b981")
    b.e("lines.value", "each.items")
    b.e("each.item", "text.json")
    b.e("each.item", "host.json")
    b.e("text.value", "route.value")
    b.e("host.value", "route.key")
    b.e("route.case_1", "voice_a.text")
    b.e("route.case_2", "voice_b.text")
    b.e("voice_a.audio", "either.a")
    b.e("voice_b.audio", "either.b")
    b.e("either.value", "each.collect")
    b.n("join", "audio.concat", 8.4, 1, gap_ms = 300)
    b.e("each.results", "join.clips")
    b.n("episode", "output.preview", 9.4, 1)
    b.e("join.audio", "episode.value")
    b.n("title", "json.get", 2, 2.6, path = "title")
    b.e("script.json", "title.json")
    b.n(
        "cover_prompt",
        "text.template",
        3,
        2.6,
        template = "Podcast cover art for an episode called '{{values}}', bold playful illustration, two cartoon "
        "microphones, vibrant colors, no text",
    )
    b.e("title.value", "cover_prompt.values")
    b.n("cover", "image.generate", 4, 2.9, model = IMAGE, size = "1024x1024")
    b.e("cover_prompt.text", "cover.prompt")
    b.n("cover_show", "output.preview", 5, 2.9)
    b.e("cover.image", "cover_show.value")
    return b


def storyboard() -> B:
    b = B(
        "storyboard-trailer",
        "Storyboard to Trailer",
        "pipeline",
        "Pitch a movie in one line. The LLM breaks it into shots, the image model paints a keyframe for each, "
        "and the video model brings every keyframe to life. Plays back as a trailer.",
        tags = ["video", "image", "for each"],
        seed = 1985,
    ).need("chat", "image", "video")
    b.n(
        "pitch",
        "input.text",
        0,
        1,
        value = "A retired robot gardener secretly enters the world's biggest pumpkin contest.",
    )
    b.n(
        "shots",
        "llm.generate",
        1,
        1,
        label = "Director",
        model = CHAT,
        temperature = 0.8,
        max_tokens = 900,
        system = "You are a film director. Write 4 cinematic shot descriptions for a trailer: subject, action, camera, light.",
        prompt = "Movie: {{prompt}}",
        json_schema = {
            "type": "object",
            "properties": {
                "shots": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "string"},
                }
            },
            "required": ["shots"],
        },
    )
    b.e("pitch.text", "shots.prompt")
    b.n("list", "json.get", 2, 1, path = "shots")
    b.e("shots.json", "list.json")
    b.n("each", "logic.for_each", 3, 1, group = "shot_loop", max_items = 4)
    b.n(
        "key",
        "image.generate",
        4,
        0.6,
        group = "shot_loop",
        label = "Keyframe",
        model = IMAGE,
        size = "1344x768",
        style = "cinematic film still, anamorphic, dramatic lighting",
    )
    b.n(
        "clip",
        "video.generate",
        5,
        1,
        group = "shot_loop",
        label = "Animate",
        model = VIDEO,
        size = "1280x704",
        seconds = 4,
    )
    b.group("shot_loop", "For each shot", (3, 0.2, 3, 1.8), "#ef4444")
    b.e("list.value", "each.items")
    b.e("each.item", "key.prompt")
    b.e("each.item", "clip.prompt")
    b.e("key.image", "clip.first_frame")
    b.e("clip.video", "each.collect")
    b.n("trailer", "video.playlist", 6.4, 1)
    b.e("each.results", "trailer.videos")
    b.n("show", "output.preview", 7.4, 1)
    b.e("trailer.videos", "show.value")
    b.note(
        "n1",
        "One video renders at a time, so the shots queue up. Try fewer frames or a smaller size for a quick draft.",
        6.4,
        2,
        1.4,
        0.8,
    )
    return b


def dataset_forge() -> B:
    b = B(
        "dataset-forge",
        "Dataset Forge -> Fine-tune",
        "pipeline",
        "Make a small model talk like a pirate (or anything else): the LLM writes training examples, they become "
        "a dataset, a LoRA fine-tune runs on it, and you compare the result against the prompt. Export to GGUF at the end.",
        tags = ["training", "dataset", "fine-tune"],
        seed = 3,
    ).need("chat")
    b.n("persona", "input.text", 0, 1, value = "a cheerful pirate who explains science facts")
    b.n(
        "writer",
        "llm.generate",
        1,
        1,
        label = "Example writer",
        model = CHAT,
        temperature = 1.0,
        max_tokens = 6000,
        system = "You write diverse supervised fine-tuning examples.",
        prompt = "Write 40 training examples where the assistant answers as {{prompt}}. Vary the questions widely.",
        json_schema = {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "minItems": 10,
                    "maxItems": 60,
                    "items": {
                        "type": "object",
                        "properties": {
                            "instruction": {"type": "string"},
                            "output": {"type": "string"},
                        },
                        "required": ["instruction", "output"],
                    },
                }
            },
            "required": ["rows"],
        },
    )
    b.e("persona.text", "writer.prompt")
    b.n("rows", "json.get", 2, 1, path = "rows")
    b.e("writer.json", "rows.json")
    b.n("ds", "data.rows_to_dataset", 3, 1, name = "pirate-science")
    b.e("rows.value", "ds.rows")
    _pirate_recipe(b, 3.4)
    b.n(
        "tune",
        "train.finetune",
        4,
        1,
        base_model = "unsloth/Qwen3-0.6B",
        training_type = "lora",
        max_steps = 60,
        learning_rate = 0.0002,
        lora_r = 16,
        max_seq_length = 2048,
        format_type = "alpaca",
    )
    b.e("ds.dataset", "tune.dataset")
    b.n("ask", "input.text", 4, 2.2, value = "Why is the sky blue?")
    b.n("test", "train.chat_checkpoint", 5, 1, max_tokens = 300)
    b.e("tune.checkpoint", "test.checkpoint")
    b.e("ask.text", "test.prompt")
    b.n("answer", "output.preview", 6, 1)
    b.e("test.text", "answer.value")
    b.n("metrics", "output.preview", 5, 0)
    b.e("tune.metrics", "metrics.value")
    b.n("gguf", "train.export", 6, 2.2, format = "gguf", quant = "q4_k_m")
    b.e("tune.checkpoint", "gguf.checkpoint")
    b.note(
        "n1",
        "Training needs a GPU and stops chat models while it runs. The Data Recipe group below is the other way "
        "to make the rows: enable its Recipe Dataset step and connect it to Fine-tune instead of Rows to Dataset.",
        3,
        -0.2,
        1.6,
        0.9,
    )
    return b


PIRATE_RECIPE = {
    "model_providers": [
        {"name": "local", "provider_type": "openai", "endpoint": "", "is_local": True}
    ],
    "model_configs": [
        {
            "alias": "writer",
            "model": "",
            "provider": "local",
            "inference_parameters": {"temperature": 0.9},
        }
    ],
    "columns": [
        {
            "column_type": "sampler",
            "name": "topic",
            "drop": False,
            "sampler_type": "category",
            "params": {"values": ["physics", "chemistry", "biology", "astronomy", "weather"]},
        },
        {
            "column_type": "llm-text",
            "name": "instruction",
            "drop": False,
            "model_alias": "writer",
            "prompt": "Ask one short, curious question a kid might ask about {{ topic }}. Output only the question.",
        },
        {
            "column_type": "llm-text",
            "name": "output",
            "drop": False,
            "model_alias": "writer",
            "system_prompt": "You are a cheerful pirate who explains science facts correctly.",
            "prompt": "{{ instruction }}",
        },
    ],
    "processors": [],
}


def _pirate_recipe(b: B, row: float) -> None:
    """The Data Recipe way to write the same rows: a Recipe group, off until you enable it."""
    doc, _ = flow_from_recipe(
        {"recipe": PIRATE_RECIPE, "run": {"rows": 40}}, name = "Pirate science recipe"
    )
    _place_recipe(b, doc, 1, row)
    generate = next(n for n in b.doc["nodes"] if n["type"] == "recipe.generate")
    generate["disabled"] = True
    generate["params"]["mode"] = "full"


def _place_recipe(
    b: B,
    doc: dict,
    col: float,
    row: float,
    *,
    preview: bool = False,
) -> None:
    dx, dy = round(col * 320), round(row * 190)
    keep = [n for n in doc["nodes"] if preview or n["type"] != "output.preview"]
    for node in keep:
        node["pos"] = [node["pos"][0] + dx, node["pos"][1] + dy]
        b.doc["nodes"].append(node)
    ids = {n["id"] for n in keep}
    b.doc["edges"].extend(e for e in doc["edges"] if e["source"] in ids and e["target"] in ids)
    for group in doc["groups"]:
        x, y, w, h = group["rect"]
        b.doc["groups"].append({**group, "rect": [x + dx, y + dy, w, h]})
    for note in doc["notes"]:
        x, y, w, h = note["rect"]
        b.doc["notes"].append({**note, "rect": [x + dx, y + dy, w, h]})


LEARNING_RECIPES = Path(__file__).with_name("learning_recipes")
# Recipe Studio's learning recipes (frontend/src/features/data-recipes/learning-recipes), same files.
LEARNING = (
    ("text-to-sql", "Text to SQL", "Generate SQL tasks and runnable SQL outputs with prompt-driven generation."),
    ("instruction-from-answer", "Instruction from Answer", "Use seed answer columns to generate high-quality instruction targets."),
    ("text-to-python", "Text to Python", "Generate instruction-to-code data with category sampling and LLM judging."),
    ("structured-outputs-jinja", "Structured Outputs + Jinja Expressions", "Support ticket triage with structured JSON outputs and Jinja conditionals."),
    ("conversation", "Multi-turn Conversation", "Sample a domain and topic, then write a whole user and assistant conversation as structured JSON."),
    ("pdf-grounded-qa", "PDF Document QA", "Build grounded question-answer examples from PDF chunks."),
    ("ocr-document-extraction", "OCR Document Extraction", "Use image context to generate OCR-style document extraction output."),
    ("github-support-bot", "GitHub Crawler", "Crawl real GitHub issues and PRs and turn each thread into a {User, Assistant} training pair."),
    ("books-and-documents", "Books & Documents to Text", "Extract ebooks, PDFs and Office files into a text dataset for continued pretraining."),
)  # fmt: skip


def learning_recipe(stem: str, title: str, description: str) -> B:
    payload = json.loads((LEARNING_RECIPES / f"{stem}.json").read_text())
    b = B(
        f"recipe-{stem}",
        title,
        "pipeline",
        description + " A Data Recipe as a group: double-click a column to edit it, select Recipe "
        "Dataset for its models, seed and rows.",
        tags = ["data", "data recipe", "dataset"],
    )
    doc, _ = flow_from_recipe(payload, name = title)
    _place_recipe(b, doc, 0, 0, preview = True)
    b.doc["meta"]["learning_recipe"] = stem
    return b


def _learning_builders() -> list:
    return [functools.partial(learning_recipe, *entry) for entry in LEARNING]


def prompt_tournament() -> B:
    b = B(
        "prompt-tournament",
        "Prompt Tournament",
        "pipeline",
        "Six prompt variants fight for the crown. Each one is painted, a vision model critiques every image, a "
        "judge LLM picks the winner, and the winning prompt gets a final high-quality render.",
        tags = ["image", "vision", "for each", "evaluation"],
        seed = 6,
    ).need("chat", "vision", "image")
    b.n("idea", "input.text", 0, 1, value = "a lighthouse made of candy on a stormy sea")
    b.n(
        "variants",
        "llm.generate",
        1,
        1,
        label = "Prompt writer",
        model = CHAT,
        temperature = 1.0,
        max_tokens = 1200,
        system = "You write rich, specific text-to-image prompts in very different styles.",
        prompt = "Write 6 different image prompts for: {{prompt}}",
        json_schema = {
            "type": "object",
            "properties": {
                "prompts": {
                    "type": "array",
                    "minItems": 6,
                    "maxItems": 6,
                    "items": {"type": "string"},
                }
            },
            "required": ["prompts"],
        },
    )
    b.e("idea.text", "variants.prompt")
    b.n("list", "json.get", 2, 1, path = "prompts")
    b.e("variants.json", "list.json")
    b.n("each", "logic.for_each", 3, 1, group = "t_loop", max_items = 6)
    b.n("paint", "image.generate", 4, 0.6, group = "t_loop", model = IMAGE, size = "768x768", steps = 4)
    b.n(
        "ask",
        "text.template",
        4,
        1.6,
        group = "t_loop",
        template = 'Score 1-10 how beautiful this image is and how well it matches: "{{values}}". '
        "Reply with the score first, then one sentence.",
    )
    b.n("critic", "llm.vision", 5, 1, group = "t_loop", model = VISION, max_tokens = 120)
    b.n(
        "pair",
        "text.template",
        6,
        1,
        group = "t_loop",
        template = "PROMPT: {{values.0}}\nCRITIQUE: {{values.1}}",
    )
    b.group("t_loop", "For each variant", (3, 0.2, 4, 2.2), "#ec4899")
    b.e("list.value", "each.items")
    b.e("each.item", "paint.prompt")
    b.e("each.item", "ask.values")
    b.e("paint.image", "critic.image")
    b.e("ask.text", "critic.question")
    b.e("each.item", "pair.values")
    b.e("critic.text", "pair.values")
    b.e("pair.text", "each.collect")
    b.n(
        "judge",
        "llm.generate",
        7.4,
        1,
        label = "Judge",
        model = CHAT,
        temperature = 0.2,
        max_tokens = 600,
        prompt = "Here are prompts with critiques of their images:\n{{context}}\nPick the single best prompt.",
        json_schema = {
            "type": "object",
            "properties": {"winner": {"type": "string"}, "why": {"type": "string"}},
            "required": ["winner", "why"],
        },
    )
    b.e("each.results", "judge.context")
    b.n("winner", "json.get", 8.4, 1, path = "winner")
    b.e("judge.json", "winner.json")
    b.n(
        "final",
        "image.generate",
        9.4,
        1,
        label = "Champion render",
        model = IMAGE,
        size = "1344x768",
        steps = 8,
    )
    b.e("winner.value", "final.prompt")
    b.n("show", "output.preview", 10.4, 1)
    b.e("final.image", "show.value")
    b.n("all", "output.preview", 8.4, 2.2, label = "Contestants")
    b.e("each.results", "all.value")
    return b


def sticker_pack() -> B:
    b = B(
        "sticker-pack",
        "Sticker Pack",
        "pipeline",
        "Design a character once, then get eight expression stickers with transparent backgrounds, laid out on a "
        "sheet and saved to your Library.",
        tags = ["image", "edit", "for each", "remove background"],
        seed = 12,
    ).need("image")
    b.n("who", "input.text", 0, 1, value = "a round orange cat wearing a tiny green frog hat")
    b.n(
        "base",
        "image.generate",
        1,
        1,
        label = "Character",
        model = IMAGE,
        size = "1024x1024",
        style = "cute sticker illustration, thick white outline, flat colors, plain light grey background",
        seed = 21,
    )
    b.e("who.text", "base.prompt")
    b.n(
        "moods",
        "input.list",
        1,
        2.2,
        items = [
            "happy",
            "laughing",
            "crying",
            "angry",
            "sleepy",
            "surprised",
            "in love",
            "thumbs up",
        ],
    )
    b.n("each", "logic.for_each", 2.2, 1.6, group = "s_loop", max_items = 8)
    b.n(
        "how",
        "text.template",
        3.2,
        1.2,
        group = "s_loop",
        template = "Make the same character look {{values}}, same style and outline, plain light grey background",
    )
    b.n("edit", "image.edit", 4.2, 1.6, group = "s_loop", model = IMAGE)
    b.n("cut", "image.remove_bg", 5.2, 1.6, group = "s_loop", subject = "the character", trim = True)
    b.group("s_loop", "For each mood", (2.2, 0.8, 4, 1.6), "#f59e0b")
    b.e("moods.items", "each.items")
    b.e("each.item", "how.values")
    b.e("base.image", "edit.image")
    b.e("how.text", "edit.instruction")
    b.e("edit.image", "cut.image")
    b.e("cut.image", "each.collect")
    b.n(
        "sheet",
        "image.grid",
        6.6,
        1.2,
        columns = 4,
        gap = 24,
        background = "#ffffff",
        caption_style = "subtitle",
        cell_size = 384,
    )
    b.e("each.results", "sheet.images")
    b.e("moods.items", "sheet.captions")
    b.n("show", "output.preview", 7.6, 1)
    b.e("sheet.image", "show.value")
    b.n("save", "output.library", 7.6, 2, folder = "Stickers")
    b.e("each.results", "save.value")
    return b


BACKING = {"repo": "audio-cpp/audio.cpp-gguf/Stable-Audio-3-Small-Music-GGUF", "variant": None}


def meme_maker() -> B:
    b = B(
        "meme-maker",
        "Meme Maker",
        "pipeline",
        "Give it a topic: the LLM writes the joke, the image model draws the scene, and the captions go on "
        "top and bottom.",
        tags = ["image", "text overlay", "fun"],
        seed = 51,
    ).need("chat", "image")
    b.n("topic", "input.text", 0, 1, value = "when the code works on the first try")
    b.n(
        "joke",
        "llm.generate",
        1,
        1,
        label = "Joke writer",
        model = CHAT,
        temperature = 0.9,
        max_tokens = 300,
        prompt = "Write a funny two-line meme about: {{prompt}}",
        json_schema = {
            "type": "object",
            "properties": {
                "scene": {"type": "string"},
                "top": {"type": "string"},
                "bottom": {"type": "string"},
            },
            "required": ["scene", "top", "bottom"],
        },
    )
    b.e("topic.text", "joke.prompt")
    for i, part in enumerate(("scene", "top", "bottom")):
        b.n(part, "json.get", 2, 0.4 + i * 0.8, path = part)
        b.e("joke.json", f"{part}.json")
    b.n("draw", "image.generate", 3, 0.4, model = IMAGE, size = "1024x1024", style = "photo, no text")
    b.e("scene.value", "draw.prompt")
    b.n("top_text", "image.text", 4, 0.8, label = "Top caption", size = 72, position = "top", outline = 4)
    b.n(
        "bottom_text",
        "image.text",
        5,
        1.2,
        label = "Bottom caption",
        size = 72,
        position = "bottom",
        outline = 4,
    )
    b.e("draw.image", "top_text.image")
    b.e("top.value", "top_text.text")
    b.e("top_text.image", "bottom_text.image")
    b.e("bottom.value", "bottom_text.text")
    b.n("show", "output.preview", 6, 1.2)
    b.e("bottom_text.image", "show.value")
    return b


def postcard() -> B:
    b = B(
        "postcard",
        "Postcard",
        "pipeline",
        "Paint a scene and a paper background, warm up the photo, place it on the card and write the greeting.",
        tags = ["image", "composite", "adjust", "text overlay"],
        seed = 52,
    ).need("image")
    b.n("place", "input.text", 0, 1, value = "a sunny seaside town with colourful houses")
    b.n("scene", "image.generate", 1, 0.5, label = "Photo", model = IMAGE, size = "1024x768")
    b.e("place.text", "scene.prompt")
    b.n(
        "paper",
        "image.generate",
        1,
        1.6,
        label = "Card",
        model = IMAGE,
        size = "1344x768",
        prompt = "blank vintage postcard paper, cream, subtle texture, empty, no text",
    )
    b.n("warm", "image.adjust", 2, 0.5, saturation = 1.25, contrast = 1.1, brightness = 1.05)
    b.e("scene.image", "warm.image")
    b.n("card", "image.composite", 3, 1, x = 0.42, y = 0.5, scale = 0.7)
    b.e("paper.image", "card.image")
    b.e("warm.image", "card.overlay")
    b.n(
        "greeting",
        "image.text",
        4,
        1,
        text = "Greetings from the seaside!",
        size = 56,
        color = "#7c2d12",
        position = "bottom-right",
        outline = 0,
    )
    b.e("card.image", "greeting.image")
    b.n("show", "output.preview", 5, 1)
    b.e("greeting.image", "show.value")
    return b


def social_clip() -> B:
    b = B(
        "social-clip",
        "Social Clip",
        "pipeline",
        "Cut the best ten seconds of your video, speed it up a little and lay a fresh music track under it.",
        tags = ["video", "edit", "music"],
        seed = 53,
    )
    b.doc["requirements"]["models"].append(
        {
            "ref": "band",
            "kind": "audio",
            "note": "Small Stable Audio music model; any music model works.",
            "optional": False,
            **BACKING,
        }
    )
    b.n("clip", "input.video", 0, 0.5, label = "Your video")
    b.n("vibe", "input.text", 0, 1.6, value = "upbeat lo-fi hip hop beat, warm, 90 bpm")
    b.n("track", "audio.music", 1, 1.6, model = BACKING, mode = "song", duration_s = 12)
    b.e("vibe.text", "track.prompt")
    b.n("cut", "video.edit", 2, 1, start_s = 0, end_s = 10, speed = 1.25)
    b.e("clip.video", "cut.video")
    b.e("track.audio", "cut.audio")
    b.n("show", "output.preview", 3, 1)
    b.e("cut.video", "show.value")
    b.note(
        "tip",
        "Upload a video in the first block, set where the cut starts and ends, then Run.",
        0,
        2.6,
        1.6,
        0.6,
    )
    return b


def photo_restyle() -> B:
    b = B(
        "photo-restyle",
        "Photo Restyle",
        "pipeline",
        "Pick a photo and say how to change it: the image model edits it to match, keeping the subject, "
        "then the result is upscaled 2x and saved to your Library.",
        tags = ["image", "edit", "upscale"],
        seed = 41,
    ).need("image")
    b.n("photo", "input.image", 0, 1, label = "Your photo")
    b.n(
        "how",
        "input.text",
        0,
        2,
        label = "Instruction",
        value = "Turn this into a soft watercolor illustration, keep the people and their poses the same",
    )
    b.n("edit", "image.edit", 1, 1, model = IMAGE)
    b.e("photo.image", "edit.image")
    b.e("how.text", "edit.instruction")
    b.n("big", "image.upscale", 2, 1, factor = "2")
    b.e("edit.image", "big.image")
    b.n("show", "output.preview", 3, 1)
    b.e("big.image", "show.value")
    b.n("save", "output.library", 3, 2, folder = "Restyled")
    b.e("big.image", "save.value")
    b.note(
        "n1",
        "Pick a photo on Your photo, then Run. Try: 'make it night', 'add snow', 'pixel art'.",
        1,
        2.2,
    )
    return b


def animate_photo() -> B:
    b = B(
        "animate-photo",
        "Animate a Photo",
        "pipeline",
        "Bring a still image to life: the video model starts from your photo and follows the motion you describe.",
        tags = ["video", "image to video"],
        seed = 24,
    ).need("video")
    b.n("photo", "input.image", 0, 1, label = "Your photo")
    b.n(
        "motion",
        "input.text",
        0,
        2,
        label = "Motion",
        value = "Slow camera push in, hair and leaves move gently in the wind, soft natural light",
    )
    b.n("clip", "video.generate", 1, 1, model = VIDEO, size = "1280x704", seconds = 4)
    b.e("photo.image", "clip.first_frame")
    b.e("motion.text", "clip.prompt")
    b.n("show", "output.preview", 2, 1)
    b.e("clip.video", "show.value")
    b.n("save", "output.library", 2, 2, folder = "Videos")
    b.e("clip.video", "save.value")
    b.note(
        "n1",
        "Pick a photo, then Run. Pick 704x1280 for a portrait photo. Describe motion and camera, not the scene.",
        0,
        3,
    )
    return b


def meeting_notes() -> B:
    b = B(
        "meeting-notes",
        "Meeting Notes",
        "pipeline",
        "Drop in a recording of a meeting, call or lecture. It is transcribed on your machine, then the LLM writes "
        "a summary, the decisions made and a list of action items with owners.",
        tags = ["audio", "transcribe", "summarize"],
        seed = 5,
    ).need("chat")
    b.n("recording", "input.audio", 0, 1, label = "Recording")
    b.n("words", "audio.transcribe", 1, 1)
    b.e("recording.audio", "words.audio")
    b.n(
        "notes",
        "llm.generate",
        2,
        1,
        label = "Note taker",
        model = CHAT,
        temperature = 0.2,
        max_tokens = 1500,
        system = "You take clear, factual meeting notes. Only use what the transcript says.",
        prompt = "Transcript:\n{{context}}\n\nWrite the notes.",
        json_schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "decisions": {"type": "array", "items": {"type": "string"}},
                "action_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string"},
                            "task": {"type": "string"},
                            "due": {"type": "string"},
                        },
                        "required": ["owner", "task"],
                    },
                },
            },
            "required": ["summary", "decisions", "action_items"],
        },
    )
    b.e("words.text", "notes.context")
    for i, (key, label) in enumerate(
        (("summary", "Summary"), ("decisions", "Decisions"), ("action_items", "Action items"))
    ):
        b.n(key, "json.get", 3, i, path = key)
        b.e("notes.json", f"{key}.json")
        b.n(f"show_{key}", "output.preview", 4, i, label = label)
        b.e(f"{key}.value", f"show_{key}.value")
    b.n("transcript", "output.preview", 2, 2.2, label = "Transcript")
    b.e("words.text", "transcript.value")
    return b


def translate_and_dub() -> B:
    b = B(
        "translate-and-dub",
        "Translate and Dub",
        "pipeline",
        "Record yourself in one language and hear it in another, in your own voice: the clip is transcribed, "
        "translated by the LLM, and spoken by a voice-cloning model that uses your clip as the reference.",
        tags = ["audio", "transcribe", "translate", "voice clone"],
        seed = 11,
    ).need("chat", "clone")
    b.n("recording", "input.audio", 0, 1, label = "Your voice")
    b.n(
        "language",
        "input.select",
        0,
        2,
        label = "Into",
        options = [
            "Spanish",
            "French",
            "German",
            "Italian",
            "Portuguese",
            "Japanese",
            "Korean",
            "Chinese",
        ],
        value = "Spanish",
    )
    b.n("words", "audio.transcribe", 1, 1)
    b.e("recording.audio", "words.audio")
    b.n(
        "translate",
        "llm.generate",
        2,
        1,
        label = "Translator",
        model = CHAT,
        temperature = 0.2,
        max_tokens = 1500,
        system = "You are a professional translator. Keep the meaning, tone and names.",
        prompt = "Translate into {{prompt}}. Reply with only the translation.\n\n{{context}}",
    )
    b.e("language.value", "translate.prompt")
    b.e("words.text", "translate.context")
    b.n("dub", "audio.speak", 3, 1, label = "Speak in your voice", model = CLONE)
    b.e("translate.text", "dub.text")
    b.e("recording.audio", "dub.reference")
    b.e("words.text", "dub.reference_text")
    b.n("show", "output.preview", 4, 1)
    b.e("dub.audio", "show.value")
    b.n("text", "output.preview", 3, 2.2, label = "Translation")
    b.e("translate.text", "text.value")
    b.note(
        "n1",
        "A clean 10 to 30 second clip clones best. Remove Reference on Speak to use a stock voice.",
        1,
        2.2,
    )
    return b


def product_listing() -> B:
    b = B(
        "product-listing",
        "Product Listing",
        "pipeline",
        "One product photo in, a shop listing out: the vision model looks at the product, the LLM writes a title, "
        "selling points and a description, and the background is cut away for a clean product shot.",
        tags = ["vision", "image", "remove background", "writing"],
        seed = 8,
    ).need("chat", "vision")
    b.n("photo", "input.image", 0, 1, label = "Product photo")
    b.n(
        "look",
        "llm.vision",
        1,
        1,
        model = VISION,
        question = "Describe this product for a shop listing: what it is, material, colors, size cues and "
        "anything notable. Facts only.",
        max_tokens = 400,
    )
    b.e("photo.image", "look.image")
    b.n(
        "copy",
        "llm.generate",
        2,
        1,
        label = "Copywriter",
        model = CHAT,
        temperature = 0.7,
        max_tokens = 800,
        system = "You write honest, punchy e-commerce listings. Never invent specs the description does not give.",
        prompt = "Product:\n{{context}}\n\nWrite the listing.",
        json_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "bullets": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 5,
                    "items": {"type": "string"},
                },
                "description": {"type": "string"},
            },
            "required": ["title", "bullets", "description"],
        },
    )
    b.e("look.text", "copy.context")
    b.n("listing", "output.preview", 3, 1, label = "Listing")
    b.e("copy.json", "listing.value")
    b.n("cut", "image.remove_bg", 1, 2.2, subject = "the product", trim = True)
    b.e("photo.image", "cut.image")
    b.n("shot", "output.preview", 2, 2.2, label = "Product shot")
    b.e("cut.image", "shot.value")
    b.n("save", "output.library", 3, 2.2, folder = "Products")
    b.e("cut.image", "save.value")
    return b


def song_maker() -> B:
    b = B(
        "song-maker",
        "Song Maker",
        "pipeline",
        "Give a theme, get a song: the LLM writes lyrics and a style, the music model sings them, and the image "
        "model paints the cover.",
        tags = ["audio", "music", "image"],
        seed = 77,
    ).need("chat", "music", optional = ("image",))
    b.n("theme", "input.text", 0, 1, value = "a sloth who finally wins a race")
    b.n(
        "writer",
        "llm.generate",
        1,
        1,
        label = "Songwriter",
        model = CHAT,
        temperature = 0.9,
        max_tokens = 900,
        system = "You write catchy song lyrics with [verse] and [chorus] tags, two verses and a chorus.",
        prompt = "Write a song about: {{prompt}}",
        json_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "style": {
                    "type": "string",
                    "description": "genre, mood, instruments, tempo, voice",
                },
                "lyrics": {"type": "string"},
            },
            "required": ["title", "style", "lyrics"],
        },
    )
    b.e("theme.text", "writer.prompt")
    for i, key in enumerate(("style", "lyrics", "title")):
        b.n(key, "json.get", 2, i, path = key)
        b.e("writer.json", f"{key}.json")
    b.n("song", "audio.music", 3, 0.5, model = MUSIC, mode = "song", duration_s = 60)
    b.e("style.value", "song.prompt")
    b.e("lyrics.value", "song.lyrics")
    b.n("play", "output.preview", 4, 0.5)
    b.e("song.audio", "play.value")
    b.n(
        "cover_prompt",
        "text.template",
        3,
        2,
        template = "Album cover for a song called '{{values}}', bold graphic illustration, vivid colors, no text",
    )
    b.e("title.value", "cover_prompt.values")
    b.n("cover", "image.generate", 4, 2, model = IMAGE, size = "1024x1024")
    b.e("cover_prompt.text", "cover.prompt")
    b.n("cover_show", "output.preview", 5, 2)
    b.e("cover.image", "cover_show.value")
    return b


def long_form_writer() -> B:
    b = B(
        "long-form-writer",
        "Long-form Writer",
        "pipeline",
        "Write an article longer than one reply can hold: the LLM plans an outline, writes each section with the "
        "outline in view, and the sections are joined into one document.",
        tags = ["writing", "for each"],
        seed = 19,
    ).need("chat")
    b.n("topic", "input.text", 0, 1, value = "A beginner's guide to growing tomatoes on a balcony")
    b.n(
        "plan",
        "llm.generate",
        1,
        1,
        label = "Outline",
        model = CHAT,
        temperature = 0.6,
        max_tokens = 800,
        system = "You plan clear, well-structured articles.",
        prompt = "Plan an article about: {{prompt}}. Each section is one line: its heading, then what it covers.",
        json_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "sections": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 6,
                    "items": {"type": "string"},
                },
            },
            "required": ["title", "sections"],
        },
    )
    b.e("topic.text", "plan.prompt")
    b.n("title", "json.get", 2, 0, path = "title")
    b.n("sections", "json.get", 2, 1, path = "sections")
    b.e("plan.json", "title.json")
    b.e("plan.json", "sections.json")
    b.n("each", "logic.for_each", 3, 1, group = "w_loop", max_items = 6)
    b.n(
        "write",
        "llm.generate",
        4,
        1,
        group = "w_loop",
        label = "Write section",
        model = CHAT,
        temperature = 0.7,
        max_tokens = 900,
        prompt = "Article outline:\n{{context}}\n\nWrite only this section, about 250 words of markdown, starting "
        "with its heading as '## ':\n{{prompt}}",
    )
    b.group("w_loop", "For each section", (3, 0.6, 2, 1.4), "#0ea5e9")
    b.e("sections.value", "each.items")
    b.e("each.item", "write.prompt")
    b.e("sections.value", "write.context")
    b.e("write.text", "each.collect")
    b.n("body", "list.join", 5.4, 1, separator = "\\n\\n")
    b.e("each.results", "body.items")
    b.n("doc", "text.template", 6.4, 1, template = "# {{values.0}}\n\n{{values.1}}")
    b.e("title.value", "doc.values")
    b.e("body.text", "doc.values")
    b.n("show", "output.preview", 7.4, 1)
    b.e("doc.text", "show.value")
    return b


def notebook_template(
    id: str,
    name: str,
    notebook: str,
    description: str,
    *,
    dataset: str,
    tune: dict[str, Any],
    export: Optional[dict[str, Any]],
    ask: Optional[str] = None,
    tags: list[str],
) -> B:
    """A Unsloth notebook as a flow: Hub dataset -> Fine-tune -> (chat test) -> (export), using the notebook's own defaults."""
    b = B(id, name, "pipeline", description, tags = ["notebook", "training", *tags], seed = 3407)
    b.n("data", "data.hf_dataset", 0, 1, repo = dataset, split = "train", subset = "")
    b.n("tune", "train.finetune", 1, 1, **{"training_type": "lora", **tune})
    b.e("data.dataset", "tune.dataset")
    b.n("metrics", "output.preview", 2, 0)
    b.e("tune.metrics", "metrics.value")
    if ask:
        b.n("ask", "input.text", 1, 2.2, value = ask)
        b.n("test", "train.chat_checkpoint", 2, 1, max_tokens = 300)
        b.e("tune.checkpoint", "test.checkpoint")
        b.e("ask.text", "test.prompt")
        b.n("answer", "output.preview", 3, 1)
        b.e("test.text", "answer.value")
    if export:
        b.n("export", "train.export", 3 if ask else 2, 2.2 if ask else 1, **export)
        b.e("tune.checkpoint", "export.checkpoint")
    b.note(
        "n1",
        f"Based on the Unsloth notebook {notebook}: {COLAB_NB}{notebook}. Training needs a GPU and stops chat "
        "models while it runs.",
        0,
        -0.2,
        1.8,
        0.9,
    )
    return b


def text_sft() -> B:
    return notebook_template(
        "notebook-text-sft",
        "Notebook: Llama 3.1 Alpaca fine-tune",
        "Llama3.1_(8B)-Alpaca.ipynb",
        "The Llama 3.1 (8B) Alpaca notebook as a flow: LoRA fine-tune on Alpaca, ask the result a question, export to GGUF.",
        dataset = "unsloth/alpaca-cleaned",
        tune = {
            "base_model": "unsloth/Llama-3.1-8B",
            "max_steps": 60,
            "learning_rate": 0.0002,
            "lora_r": 16,
            "max_seq_length": 2048,
            "format_type": "alpaca",
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 5,
        },
        ask = "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8",
        export = {"format": "gguf", "quant": "q4_k_m"},
        tags = ["sft"],
    )


def vision_sft() -> B:
    return notebook_template(
        "notebook-vision-sft",
        "Notebook: Qwen2.5-VL LaTeX OCR fine-tune",
        "Qwen2.5_VL_(7B)-Vision.ipynb",
        "The Qwen2.5-VL (7B) vision notebook as a flow: teach the model to write LaTeX from images of equations, then save the adapter.",
        dataset = "unsloth/LaTeX_OCR",
        tune = {
            "base_model": "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
            "task": "vision",
            "max_steps": 30,
            "learning_rate": 0.0002,
            "lora_r": 16,
            "max_seq_length": 2048,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 5,
        },
        export = {"format": "lora"},
        tags = ["vision"],
    )


def tts_sft() -> B:
    return notebook_template(
        "notebook-tts-sft",
        "Notebook: Orpheus TTS voice fine-tune",
        "Orpheus_(3B)-TTS.ipynb",
        "The Orpheus (3B) TTS notebook as a flow: fine-tune a voice on a speech dataset, then save the adapter.",
        dataset = "Etherll/kaira",
        tune = {
            "base_model": "unsloth/orpheus-3b-0.1-ft",
            "load_in_4bit": False,
            "task": "audio",
            "max_steps": 60,
            "learning_rate": 0.0002,
            "lora_r": 64,
            "lora_alpha": 64,
            "max_seq_length": 2048,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 5,
        },
        export = {"format": "lora"},
        tags = ["audio", "tts"],
    )


def embedding_sft() -> B:
    return notebook_template(
        "notebook-embedding",
        "Notebook: EmbeddingGemma retrieval fine-tune",
        "EmbeddingGemma_(300M).ipynb",
        "The EmbeddingGemma (300M) notebook as a flow: fine-tune an embedding model for question-to-passage "
        "retrieval, then export to GGUF.",
        # Studio's embedding trainer loads whole datasets, so this swaps the notebook's 13 GB set for a 40 MB one of the same shape.
        dataset = "sentence-transformers/natural-questions",
        tune = {
            "base_model": "unsloth/embeddinggemma-300m",
            "task": "embedding",
            "max_steps": 30,
            "learning_rate": 0.00002,
            "lora_r": 32,
            "lora_alpha": 64,
            "max_seq_length": 1024,
            "batch_size": 64,
            "gradient_accumulation_steps": 2,
            "dataset_slice_end": 19999,
        },
        export = {"format": "gguf", "quant": "q4_k_m"},
        tags = ["embedding"],
    )


def model_arena() -> B:
    b = B(
        "model-arena",
        "Model Arena",
        "pipeline",
        "Two local chat models answer the same prompts, a judge scores every answer, and the scores land in a "
        "table and a bar chart with each model's lead over the other.",
        tags = ["evaluation", "compare", "table", "chart", "for each"],
        seed = 41,
    ).need("chat")
    b.doc["requirements"]["models"].append(
        {
            "ref": "rival",
            "kind": "chat",
            "note": "The second contestant; any chat model works.",
            "optional": False,
            **RIVAL,
        }
    )
    b.n(
        "prompts",
        "input.list",
        0,
        1,
        items = [
            "Explain why the sky is blue to a ten year old.",
            "Write a haiku about debugging at 3am.",
            "Give three tips for a first marathon.",
        ],
    )
    b.n("each", "logic.for_each", 1, 1, group = "arena_loop", max_items = 8)
    b.e("prompts.items", "each.items")
    for nid, row, label, model in (("a", 0.5, "Qwen3 4B", CHAT), ("b", 1.5, "Gemma 3 4B", RIVAL)):
        b.n(
            nid,
            "llm.generate",
            2,
            row,
            group = "arena_loop",
            label = label,
            model = model,
            temperature = 0.3,
            max_tokens = 400,
        )
        b.e("each.item", f"{nid}.prompt")
    b.n(
        "pair",
        "text.template",
        3,
        1,
        group = "arena_loop",
        template = "PROMPT: {{values.0}}\nQwen3 4B: {{values.1}}\nGemma 3 4B: {{values.2}}",
    )
    b.e("each.item", "pair.values")
    b.e("a.text", "pair.values")
    b.e("b.text", "pair.values")
    b.e("pair.text", "each.collect")
    b.group("arena_loop", "For each prompt", (1, 0.2, 3, 2.2), "#22c55e")
    b.n(
        "judge",
        "llm.generate",
        4.4,
        1,
        label = "Judge",
        model = CHAT,
        temperature = 0.0,
        max_tokens = 1200,
        system = "You are a strict, fair judge of chat answers.",
        prompt = "Score each model's answer to each prompt from 1 to 10 for helpfulness and accuracy:\n{{context}}",
        json_schema = {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "model": {"type": "string", "enum": ["Qwen3 4B", "Gemma 3 4B"]},
                            "score": {"type": "number"},
                        },
                        "required": ["prompt", "model", "score"],
                    },
                }
            },
            "required": ["rows"],
        },
    )
    b.e("each.results", "judge.context")
    b.n("rows", "json.get", 5.4, 1, path = "rows")
    b.e("judge.json", "rows.json")
    b.n("scores", "table.concat", 6.4, 1, label = "Every score")
    b.e("rows.value", "scores.tables")
    b.n(
        "avg",
        "table.aggregate",
        7.4,
        1,
        label = "Mean per model",
        group_by = ["model"],
        metrics = ["score"],
        stat = "mean",
    )
    b.e("scores.table", "avg.table")
    b.n(
        "lead",
        "table.compare",
        8.4,
        0.4,
        label = "Lead over Qwen3",
        key = ["model"],
        metric = "score",
        baseline_row = "Qwen3 4B",
    )
    b.e("avg.table", "lead.table")
    b.n("bars", "table.chart", 8.4, 1.6, type = "bar", x = "model", y = ["score"])
    b.e("avg.table", "bars.table")
    b.n("show_scores", "output.preview", 7.4, 2.2, label = "Every score")
    b.e("scores.table", "show_scores.value")
    b.n("show_lead", "output.preview", 9.4, 0.4)
    b.e("lead.table", "show_lead.value")
    b.n("show_chart", "output.preview", 9.4, 1.6)
    b.e("bars.chart", "show_chart.value")
    b.note(
        "tip",
        "Swap either contestant's model or add prompts to the list, then Run.",
        0,
        2.2,
        1.6,
        0.6,
    )
    return b


def smallville_lite() -> B:
    b = B(
        "smallville-lite",
        "Smallville Lite",
        "world",
        "A small town whose six villagers live their own day: they remember, plan, walk between places and chat with "
        "each other, so news spreads without you. After Park et al. 2023, Generative Agents.",
        tags = ["world", "game", "npc", "memory", "agent"],
        seed = 214,
    ).need("chat")
    places = [
        {
            "id": "hobbs_cafe",
            "name": "Hobbs Cafe",
            "x": 20,
            "y": 6,
            "w": 8,
            "h": 5,
            "on_enter_text": "Smells of fresh coffee.",
        },
        {
            "id": "johnson_park",
            "name": "Johnson Park",
            "x": 5,
            "y": 21,
            "w": 10,
            "h": 7,
            "on_enter_text": "Birds and benches.",
        },
        {
            "id": "the_market",
            "name": "The Market",
            "x": 33,
            "y": 21,
            "w": 10,
            "h": 6,
            "on_enter_text": "Stalls and chatter.",
        },
        {
            "id": "lin_house",
            "name": "Lin family home",
            "x": 5,
            "y": 5,
            "w": 7,
            "h": 5,
            "on_enter_text": "A cosy blue house.",
        },
        {
            "id": "moreno_house",
            "name": "Moreno house",
            "x": 36,
            "y": 5,
            "w": 7,
            "h": 5,
            "on_enter_text": "A tidy garden.",
        },
    ]

    def npc(
        name: str,
        at: str,
        persona: str,
        routine: str,
        relationships: dict,
        memories: list,
        dx: int = 0,
    ) -> dict:
        p = next(x for x in places if x["id"] == at)
        return {
            "name": name,
            "persona": persona,
            "greeting": f"Hi, I'm {name.split()[0]}.",
            "routine": routine,
            "relationships": relationships,
            "seed_memories": memories,
            "x": p["x"] + p["w"] // 2 + dx,
            "y": p["y"] + p["h"] // 2,
        }

    world = {
        "title": "Smallville Lite",
        "seed": 214,
        "biome": "town",
        "width": 48,
        "height": 34,
        "intro_text": "It is February 14 in Smallville. Isabella is planning something at Hobbs Cafe tonight. "
        "Walk around, talk to people, and watch the news spread.",
        "player": {"name": "Pip"},
        "npcs": [
            npc(
                "Isabella Rodriguez",
                "hobbs_cafe",
                "Owner of Hobbs Cafe; warm, outgoing and loves bringing people together.",
                "Opens the cafe at 8, serves customers all day, closes at 8 in the evening.",
                {
                    "Maria Lopez": "a regular at the cafe and a good friend",
                    "Tom Moreno": "sells her coffee beans",
                },
                [
                    {
                        "text": "I am hosting a Valentine's Day party at Hobbs Cafe on February 14 from 5pm to 7pm",
                        "importance": 8,
                    },
                    {
                        "text": "I want to invite as many people as I can to my party",
                        "importance": 6,
                    },
                ],
            ),
            npc(
                "Maria Lopez",
                "lin_house",
                "A physics student who streams games in the evening; curious and chatty.",
                "Studies in the morning, has lunch at Hobbs Cafe, walks in the park in the afternoon.",
                {
                    "Klaus Mueller": "a classmate she has a quiet crush on",
                    "Isabella Rodriguez": "her friend at the cafe",
                },
                [{"text": "I have a crush on Klaus Mueller", "importance": 6}],
            ),
            npc(
                "Klaus Mueller",
                "lin_house",
                "A sociology student writing a paper on gentrification; thoughtful and a little shy.",
                "Reads at home in the morning, takes notes in the park, shops at the market before dinner.",
                {"Maria Lopez": "a friendly classmate"},
                [{"text": "My research paper on gentrification is due next week", "importance": 5}],
                dx = -2,
            ),
            npc(
                "Tom Moreno",
                "the_market",
                "Runs the grocery stall at the market; grumbles about prices but helps everyone.",
                "Sets up his stall at 7, sells until 6, then goes home to the Moreno house.",
                {
                    "Isabella Rodriguez": "buys his coffee beans",
                    "Sam Moore": "an old friend he argues politics with",
                },
                [
                    {
                        "text": "Sam Moore is running for mayor and I am not sure I will vote for him",
                        "importance": 5,
                    }
                ],
            ),
            npc(
                "Sam Moore",
                "moreno_house",
                "A retired shop owner running for local mayor; friendly, talks to everyone about his plans.",
                "Walks around town all day to talk to voters, ending at the park in the evening.",
                {"Tom Moreno": "an old friend who doubts his campaign"},
                [
                    {
                        "text": "I am running for mayor of Smallville and want to meet every resident",
                        "importance": 7,
                    }
                ],
            ),
            npc(
                "Ayesha Khan",
                "johnson_park",
                "A literature student writing a senior thesis on Shakespeare; quiet and observant.",
                "Writes in the park in the morning, works at Hobbs Cafe in the afternoon.",
                {"Isabella Rodriguez": "her boss at the cafe"},
                [{"text": "I work afternoon shifts at Hobbs Cafe", "importance": 4}],
            ),
        ],
        "items": [
            {
                "id": "chocolates",
                "name": "Box of chocolates",
                "description": "Perfect for a Valentine's party.",
                "x": 38,
                "y": 23,
            },
        ],
        "quests": [
            {
                "id": "party_treats",
                "title": "Treats for the party",
                "description": "Isabella needs something sweet for the Valentine's Day party.",
                "giver": "isabella_rodriguez",
                "requires_item": "chocolates",
                "reward_text": "Isabella beams and saves you the best seat.",
            }
        ],
        "regions": places,
        "vars": {"party_attendance": 0},
        "simulation": {
            "minutes_per_second": 2,
            "start_minute": 480,
            "chat_cooldown_min": 120,
            "calls_per_minute": 4,
        },
    }
    b.n("world", "world.spec", 0, 1, world = world)
    b.n("play", "world.play", 1, 1)
    b.e("world.world", "play.world")
    b.n(
        "reply",
        "world.npc_reply",
        2,
        0.2,
        label = "Talk (live)",
        model = CHAT,
        max_tokens = 160,
        rules = "Stay in character. Keep replies under 50 words. Talk about what you remember and plan. If you decide "
        "to go to Isabella's party, set party_attendance to one more than its current value.",
    )
    b.n("plan", "world.npc_plan", 2, 1.2, label = "Daily plans (live)", model = CHAT)
    b.n("chat", "world.npc_chat", 2, 2.2, label = "Villager chats (live)", model = CHAT)
    b.note(
        "about",
        'Village life after Park et al. 2023, "Generative Agents: Interactive Simulacra of Human Behavior" '
        "(arXiv:2304.03442). Each villager keeps a memory stream; replies, plans and chats recall what matters by "
        "recency, importance and relevance, and record new memories in the same call.\n\n"
        "Play: one game day is 12 real minutes. Villagers plan their day once, walk between places and chat when they "
        "meet, at most 4 background model calls a minute; time stops while you talk. Everything runs on the local chat "
        "model (and the local embedding model when one is loaded). Watch whether word of Isabella's party spreads.",
        0,
        2.2,
        1.8,
        1.4,
    )
    return b


def village_rivals() -> B:
    b = B(
        "village-rivals",
        "Village Rivals",
        "world",
        "A turn-based 4X-lite on a village map: you and two rival villages build, claim land, trade, ally and "
        "raid, one day per turn. Each rival's strategist is one chat model call a turn that sets its posture; "
        "deterministic managers do the rest.",
        tags = ["world", "game", "strategy", "npc", "agent"],
        seed = 377,
    ).need("chat")
    towns = [
        {"id": "willow_ford", "name": "Willow Ford", "x": 3, "y": 3, "w": 10, "h": 8},
        {"id": "stonebrook", "name": "Stonebrook", "x": 34, "y": 4, "w": 10, "h": 8},
        {"id": "ember_hill", "name": "Ember Hill", "x": 18, "y": 25, "w": 10, "h": 8},
    ]
    world = {
        "title": "Village Rivals",
        "seed": 377,
        "biome": "meadow",
        "width": 48,
        "height": 36,
        "intro_text": "Three villages share one valley and not enough land. Grow Willow Ford, talk to the other "
        "leaders, and end each day when you are ready.",
        "player": {"name": "Reeve"},
        "npcs": [
            {
                "name": "Mara Quill",
                "persona": "Reeve of Willow Ford's council; a careful baker who counts every loaf.",
                "greeting": "The council trusts you to lead us, Reeve.",
                "x": 8,
                "y": 7,
            },
            {
                "name": "Bram Holt",
                "persona": "Chief of Stonebrook; a proud stone mason who respects strength and keeps his word.",
                "greeting": "Stonebrook does not bend.",
                "x": 39,
                "y": 8,
            },
            {
                "name": "Isolde Fenn",
                "persona": "Elder of Ember Hill; a shrewd trader who prefers deals to fights.",
                "greeting": "Everything has a price, friend.",
                "x": 23,
                "y": 29,
            },
        ],
        "items": [],
        "quests": [],
        "regions": [{**t, "on_enter_text": f"You enter {t['name']}."} for t in towns],
        "rivals": {
            "factions": [
                {
                    "id": "willow_ford",
                    "name": "Willow Ford",
                    "player": True,
                    "region": "willow_ford",
                    "leader": "mara_quill",
                },
                {
                    "id": "stonebrook",
                    "name": "Stonebrook",
                    "region": "stonebrook",
                    "leader": "bram_holt",
                    "persona": "Masons and miners; quick to take offence, slow to forget.",
                },
                {
                    "id": "ember_hill",
                    "name": "Ember Hill",
                    "region": "ember_hill",
                    "leader": "isolde_fenn",
                    "persona": "Merchants first; they would rather buy land than fight for it.",
                },
            ],
            "max_turns": 30,
        },
    }
    b.n("world", "world.spec", 0, 1, world = world)
    b.n("play", "world.play", 1, 1)
    b.e("world.world", "play.world")
    b.n(
        "reply",
        "world.npc_reply",
        2,
        1,
        label = "Leaders and strategists (live)",
        model = CHAT,
        max_tokens = 160,
        rules = "Stay in character. Keep replies under 50 words.",
    )
    b.note(
        "about",
        "Play: open Village Rivals in the player. Each day you get two actions (build, claim land, train, raid, "
        "or accept a deal), then End day. Each rival village makes one model call a day for its posture "
        "(economy, expansion, military, diplomacy), a target and goals; it only switches plans when the new "
        "posture differs enough. Talk to a leader to trade or ally: a deal shows up as a button only when the "
        "game can carry it out. Without a chat model the rivals still play on their own.",
        0,
        2.2,
        1.8,
        1.2,
    )
    return b


def remembering_assistant() -> B:
    b = B(
        "remembering-assistant",
        "Remembering Assistant",
        "pipeline",
        "An assistant with a memory stream (after Park et al. 2023, Generative Agents): it recalls what matters "
        "for your message by recency, importance and meaning, answers with it, and adds the answer to its memory. "
        "Each run starts from the facts in What it already knows: add a line there to teach it something.",
        tags = ["memory", "agent", "chat"],
        seed = 41,
    ).need("chat")
    b.n(
        "past",
        "input.list",
        0,
        0.4,
        label = "What it already knows",
        items = [
            "Maya is training for a half marathon in May.",
            "Maya hurt her left knee running hills last Tuesday.",
            "Maya's favourite breakfast is oatmeal with blueberries.",
            "Maya's sister Ana visits next weekend.",
            "Maya said she wants to cook more at home this month.",
        ],
    )
    b.n("seed", "memory.add", 1, 0.4, label = "Remember", model = CHAT)
    b.e("past.items", "seed.items")
    b.n(
        "message",
        "input.text",
        0,
        1.4,
        label = "Your message",
        value = "Should I run hills again tomorrow?",
    )
    b.n("recall", "memory.retrieve", 2, 1, label = "Recall", top_k = 3)
    b.e("seed.memory", "recall.memory")
    b.e("message.text", "recall.query")
    b.n(
        "answer",
        "llm.generate",
        3,
        1,
        label = "Answer",
        model = CHAT,
        temperature = 0.6,
        max_tokens = 400,
        system = "You are Maya's friendly assistant. Use what you remember when it helps; keep it short.",
        prompt = "What you remember:\n{{context}}\n\nMaya says: {{prompt}}",
    )
    b.e("recall.context", "answer.context")
    b.e("message.text", "answer.prompt")
    b.n("keep", "memory.add", 4, 1, label = "Remember the answer", model = CHAT)
    b.e("recall.memory", "keep.memory")
    b.e("answer.text", "keep.items")
    b.n("show", "output.preview", 5, 1)
    b.e("answer.text", "show.value")
    b.e("recall.records", "show.value")
    b.n("stream", "output.preview", 5, 2, label = "Updated memory")
    b.e("keep.memory", "stream.value")
    b.note(
        "tip",
        "Recall scores every memory by how recent, how important and how related it is to the message. "
        "Everything runs on the local chat and embedding models.",
        2,
        2.0,
        1.8,
        0.6,
    )
    return b


def summarise_pdf() -> B:
    b = B(
        "summarise-pdf",
        "Summarise a Document",
        "pipeline",
        "Load a PDF, Word or text file, cut it into chunks, summarise each chunk, then write one summary of the "
        "whole document.",
        tags = ["documents", "llm", "for each"],
        seed = 12,
    ).need("chat")
    b.n("doc", "input.file", 0, 1, label = "Your document")
    b.n("read", "doc.load", 1, 1)
    b.e("doc.file", "read.file")
    b.n("chunks", "text.split", 2, 1, mode = "chunks", unit = "words", chunk_size = 700, overlap = 50)
    b.e("read.text", "chunks.text")
    b.n("each", "logic.for_each", 3, 1, group = "s_loop", max_items = 32)
    b.n(
        "part",
        "llm.generate",
        4,
        1,
        group = "s_loop",
        label = "Summarise a part",
        model = CHAT,
        temperature = 0.2,
        max_tokens = 300,
        prompt = "Summarise this part of a document in 3 short bullet points:\n\n{{prompt}}",
    )
    b.group("s_loop", "For each chunk", (3, 0.4, 2.2, 1.6), "#0d9488")
    b.e("chunks.items", "each.items")
    b.e("each.item", "part.prompt")
    b.e("part.text", "each.collect")
    b.n("joined", "list.join", 5.4, 1)
    b.e("each.results", "joined.items")
    b.n(
        "summary",
        "llm.generate",
        6.4,
        1,
        label = "Whole summary",
        model = CHAT,
        temperature = 0.3,
        max_tokens = 700,
        prompt = "These are notes on each part of one document, in order:\n{{context}}\n\nWrite a clear summary "
        "of the whole document in one paragraph, then list the 5 key points.",
    )
    b.e("joined.text", "summary.context")
    b.n("show", "output.preview", 7.4, 1)
    b.e("summary.text", "show.value")
    b.note("tip", "Upload a document in the first block, then Run.", 0, 2.2, 1.6, 0.6)
    return b


def ask_documents() -> B:
    b = B(
        "ask-documents",
        "Ask Your Documents",
        "pipeline",
        "Add a document to a knowledge base (the same ones chat searches), then answer a question from the passages "
        "that match it, with the sources cited.",
        tags = ["documents", "knowledge base", "rag", "llm"],
        seed = 13,
    ).need("chat", "embedding")
    b.n("doc", "input.file", 0, 0.5, label = "Your document")
    b.n("add", "kb.add", 1, 0.5, knowledge_base = "Flows")
    b.e("doc.file", "add.file")
    b.n("question", "input.text", 0, 1.6, value = "What are the main points of this document?")
    b.n("search", "kb.search", 2, 1.2, knowledge_base = "Flows", top_k = 6)
    b.e("question.text", "search.query")
    b.e("add.knowledge_base", "search.after")
    b.n(
        "answer",
        "llm.generate",
        3,
        1.2,
        label = "Answer",
        model = CHAT,
        temperature = 0.2,
        max_tokens = 600,
        prompt = "Answer the question using only these passages, and cite them like [1].\n\n{{context}}\n\n"
        "Question: {{prompt}}",
    )
    b.e("question.text", "answer.prompt")
    b.e("search.context", "answer.context")
    b.n("show", "output.preview", 4, 1.2)
    b.e("answer.text", "show.value")
    b.note(
        "tip",
        "Upload a document, then Run. Parsing, embedding and search all run locally on Studio's own models, "
        "offline once they are downloaded. Later runs can drop the first two blocks and search what is indexed.",
        0,
        2.5,
        1.8,
        0.7,
    )
    return b


def contact_cleanup() -> B:
    b = B(
        "contact-cleanup",
        "Clean Up a Contact List",
        "pipeline",
        "Paste a messy list or spreadsheet: a local model turns it into rows, then the list blocks drop rows "
        "without an email, remove duplicates, tidy the fields and sort by name.",
        tags = ["data", "json", "transform"],
        seed = 13,
    ).need("chat")
    b.n(
        "pasted",
        "input.text",
        0,
        1,
        label = "Pasted list",
        value = "name,email,company\nAna Lima,ANA@example.com,Acme\nben ortiz,ben@example.org,\nAna Lima,"
        "ana@example.com,Acme Inc\nChris Wu,,Globex\nDee Park,dee@example.net,Initech",
    )
    b.n(
        "rows",
        "llm.generate",
        1,
        1,
        label = "Read the rows",
        model = CHAT,
        temperature = 0.0,
        max_tokens = 2048,
        system = "Copy every person in the text into rows exactly as written. Use an empty string for a missing "
        "field. Do not fix, merge or drop anything.",
        prompt = "{{prompt}}",
        json_schema = {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                            "company": {"type": "string"},
                        },
                        "required": ["name", "email", "company"],
                    },
                }
            },
            "required": ["rows"],
        },
    )
    b.e("pasted.text", "rows.prompt")
    b.n("list", "json.get", 2, 1, path = "rows")
    b.e("rows.json", "list.json")
    b.n(
        "has_email",
        "list.filter",
        3,
        1,
        label = "Has an email",
        condition = {"op": "contains", "args": [{"var": "item.email"}, "@"]},
    )
    b.e("list.value", "has_email.items")
    b.n(
        "tidy",
        "json.set",
        4,
        1,
        label = "Tidy fields",
        fields = {
            "email": {"op": "lower", "args": [{"var": "item.email"}]},
            "company": {
                "op": "if",
                "args": [{"var": "item.company"}, {"var": "item.company"}, "unknown"],
            },
        },
    )
    b.e("has_email.items", "tidy.value")
    b.n("unique", "list.dedupe", 5, 1, label = "One per email", key = {"var": "item.email"})
    b.e("tidy.value", "unique.items")
    b.n(
        "sorted",
        "list.sort",
        6,
        1,
        label = "Sort by name",
        key = {"op": "lower", "args": [{"var": "item.name"}]},
    )
    b.e("unique.items", "sorted.items")
    b.n("clean", "output.result", 7, 1, name = "contacts")
    b.e("sorted.items", "clean.value")
    b.n("dropped", "output.preview", 4, 2)
    b.e("has_email.dropped", "dropped.value")
    b.note(
        "tip",
        "Only the first block uses a model; the cleanup after it is plain list blocks, so you can change a rule "
        "and re-run in a blink.",
        0,
        2,
        1.8,
        0.7,
    )
    return b


def chat_with_documents() -> B:
    b = B(
        "chat-with-documents",
        "Chat With My Documents",
        "pipeline",
        "A chat assistant that answers from one of your knowledge bases, fully offline with a local model. Open it "
        "in the player to chat; pick the knowledge base in the Agent block.",
        tags = ["chat", "agent", "knowledge base"],
        seed = 41,
    ).need("chat")
    b.n("in", "chat.input", 0, 1, welcome = "Ask me anything about your documents.")
    b.n("memory", "chat.memory", 1, 1.6, turns = 6)
    b.e("in.history", "memory.history")
    b.n(
        "agent",
        "llm.agent",
        2,
        1,
        model = CHAT,
        system = "Answer from the knowledge base and say which document you used. If it does not have the answer, "
        "say so.",
        web_search = False,
    )
    b.e("in.message", "agent.prompt")
    b.e("memory.history", "agent.history")
    b.n("out", "chat.output", 3, 1)
    b.e("agent.text", "out.reply")
    b.note(
        "tip",
        "Pick a knowledge base in the Agent block, then press Play to chat.",
        0,
        2.4,
        1.8,
        0.6,
    )
    return b


def support_bot() -> B:
    b = B(
        "support-bot",
        "Support Bot",
        "pipeline",
        "Sorts each message (question, bug report or other), then answers in the matching style. Open it in the "
        "player to chat.",
        tags = ["chat", "classify"],
        seed = 42,
    ).need("chat")
    b.n("in", "chat.input", 0, 1, welcome = "Hi! What can I help you with today?")
    b.n("kind", "llm.classify", 1, 0.4, model = CHAT, categories = ["question", "bug report", "other"])
    b.e("in.message", "kind.text")
    b.n("memory", "chat.memory", 1, 1.6, turns = 4)
    b.e("in.history", "memory.history")
    b.n(
        "reply",
        "llm.generate",
        2,
        1,
        model = CHAT,
        temperature = 0.5,
        system = "You are a friendly support agent for a software product. The context gives the message's "
        "category (question, bug report or other) and the conversation so far. For a bug report, thank them and "
        "ask for steps to reproduce. Keep it short.",
        prompt = "New message: {{prompt}}",
    )
    b.e("kind.label", "reply.context")
    b.e("memory.text", "reply.context")
    b.e("in.message", "reply.prompt")
    b.n("out", "chat.output", 3, 1)
    b.e("reply.text", "out.reply")
    return b


def research_agent() -> B:
    b = B(
        "research-agent",
        "Research Agent",
        "pipeline",
        "Give it a question: a local model searches the web, reads what it finds and writes a short report with "
        "its sources.",
        tags = ["agent", "web search"],
        seed = 43,
    ).need("chat")
    b.n("question", "input.text", 0, 1, value = "What changed in the latest stable Python release?")
    b.n(
        "agent",
        "llm.agent",
        1,
        1,
        label = "Researcher",
        model = CHAT,
        system = "Research the question with web search. Write a short report in plain sentences, then a Sources "
        "list with the URL of every page you used. Never invent a source.",
        web_search = True,
        max_tool_calls = 6,
        temperature = 0.3,
    )
    b.e("question.text", "agent.prompt")
    b.n("report", "output.result", 2, 1, name = "report")
    b.e("agent.text", "report.value")
    b.note(
        "tip",
        "Web search is the only tool the Agent gets here; the model itself runs on your machine.",
        0,
        2,
        1.8,
        0.6,
    )
    return b


def invoice_extractor() -> B:
    b = B(
        "invoice-extractor",
        "Invoice Extractor",
        "pipeline",
        "Pulls the vendor, date, total and line items out of a pasted invoice or receipt as clean JSON.",
        tags = ["extract", "json"],
        seed = 44,
    ).need("chat")
    b.n(
        "invoice",
        "input.text",
        0,
        1,
        value = "ACME Tools Ltd\nInvoice #2041, 3 March 2026\n2 x Claw hammer @ 18.50\n1 x Tape measure @ 7.25\n"
        "Shipping 4.00\nTotal due: 48.25 GBP",
    )
    b.n(
        "extract",
        "llm.extract",
        1,
        1,
        model = CHAT,
        instructions = "Amounts are numbers without currency signs. Dates are YYYY-MM-DD.",
        json_schema = {
            "type": "object",
            "properties": {
                "vendor": {"type": "string"},
                "invoice_number": {"type": "string"},
                "date": {"type": "string"},
                "currency": {"type": "string"},
                "total": {"type": "number"},
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "number"},
                            "unit_price": {"type": "number"},
                        },
                        "required": ["description", "quantity", "unit_price"],
                    },
                },
            },
            "required": ["vendor", "date", "currency", "total", "line_items"],
        },
    )
    b.e("invoice.text", "extract.text")
    b.n("total", "json.get", 2, 0.5, path = "total")
    b.e("extract.json", "total.json")
    b.n("show_total", "output.preview", 3, 0.5)
    b.e("total.value", "show_total.value")
    b.n("data", "output.result", 2, 1.5, name = "invoice")
    b.e("extract.json", "data.value")
    b.note(
        "tip",
        'For a scan or photo, put an Image block and Describe Image ("Copy every line of this invoice") in front '
        "of Extract.",
        0,
        2,
        1.8,
        0.7,
    )
    return b


def email_triage() -> B:
    b = B(
        "email-triage",
        "Email Triage",
        "pipeline",
        "Sorts an incoming email (refund, bug report or other) and drafts a reply in the right style for each kind.",
        tags = ["classify", "switch", "email"],
        seed = 45,
    ).need("chat")
    b.n(
        "email",
        "input.text",
        0,
        1,
        value = "Hi, I was charged twice for my subscription this month. Can I get one of them refunded? Thanks, Sam",
    )
    b.n("kind", "llm.classify", 1, 0.4, model = CHAT, categories = ["refund", "bug report", "other"])
    b.e("email.text", "kind.text")
    b.n("route", "logic.switch", 2, 1, cases = ["refund", "bug report"])
    b.e("email.text", "route.value")
    b.e("kind.label", "route.key")
    replies = (
        (
            "refund",
            "case_1",
            "Apologise, confirm the refund will be checked within two working days, and ask for "
            "the order number if it is missing.",
        ),
        (
            "bug",
            "case_2",
            "Thank them, ask for the steps to reproduce, their app version and a screenshot.",
        ),
        ("other", "default", "Answer helpfully, or say who on the team will follow up."),
    )
    for row, (name, port, style) in enumerate(replies):
        b.n(
            f"reply_{name}",
            "llm.generate",
            3,
            row * 0.9 + 0.1,
            label = f"Reply: {name}",
            model = CHAT,
            temperature = 0.4,
            max_tokens = 400,
            system = f"You draft short, friendly support emails. {style} Sign off as the Support Team.",
            prompt = "Reply to this email:\n\n{{prompt}}",
        )
        b.e(f"route.{port}", f"reply_{name}.prompt")
    b.n("first", "logic.merge", 4, 1)
    b.e("reply_refund.text", "first.a")
    b.e("reply_bug.text", "first.b")
    b.e("reply_other.text", "first.c")
    b.n("draft", "output.result", 5, 1, name = "draft")
    b.e("first.value", "draft.value")
    b.n("label", "output.result", 2, 0.1, name = "category")
    b.e("kind.label", "label.value")
    return b


def daily_digest() -> B:
    b = B(
        "daily-digest",
        "Daily Digest",
        "pipeline",
        "Every morning, fetches the Hacker News front page, has a local model write a short digest and reads it "
        "aloud into your Library.",
        tags = ["automation", "schedule", "http", "audio"],
        seed = 46,
    ).need("chat", "tts")
    b.n("every_day", "trigger.schedule", 0, 1, cron = "0 8 * * *")
    b.n(
        "fetch",
        "http.request",
        1,
        1,
        label = "Front page",
        url = "https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage=15"
        "&attributesToRetrieve=title,url,points",
    )
    b.n("stories", "json.get", 2, 1, path = "hits")
    b.e("fetch.data", "stories.json")
    b.n(
        "digest",
        "llm.generate",
        3,
        1,
        model = CHAT,
        temperature = 0.4,
        max_tokens = 600,
        system = "You write a calm two-minute morning briefing. Group the stories by theme, one or two sentences "
        "each, plain prose that reads well aloud, no lists or links.",
        prompt = "Today's front page stories, with their points, are in the context.",
    )
    b.e("stories.value", "digest.context")
    b.n("text", "output.result", 4, 0.5, name = "digest")
    b.e("digest.text", "text.value")
    b.n("speak", "audio.speak", 4, 1.5, model = TTS, voice = "af_heart")
    b.e("digest.text", "speak.text")
    b.n("save", "output.library", 5, 1.5, folder = "Digests")
    b.e("speak.audio", "save.value")
    b.note(
        "tip",
        "Studio asks you to approve the schedule and hn.algolia.com before this runs. The schedule is off until "
        "you switch it on, and fires only while Studio is open. Press Run to try it once now.",
        0,
        2,
        2,
        0.8,
    )
    return b


def webhook_reply() -> B:
    b = B(
        "webhook-reply",
        "Webhook Auto-Reply",
        "pipeline",
        "Another app POSTs a message to this flow's secret URL; a local model drafts a reply you find in the run "
        "history.",
        tags = ["automation", "webhook"],
        seed = 47,
    ).need("chat")
    b.n(
        "hook",
        "trigger.webhook",
        0,
        1,
        test_payload = {
            "name": "Ana",
            "message": "Do you ship to Canada, and how long does it take?",
        },
    )
    b.n("message", "json.get", 1, 0.5, path = "message")
    b.e("hook.payload", "message.json")
    b.n("name", "json.get", 1, 1.5, path = "name")
    b.e("hook.payload", "name.json")
    b.n(
        "reply",
        "llm.generate",
        2,
        1,
        model = CHAT,
        temperature = 0.4,
        max_tokens = 400,
        system = "You answer customer messages for a small online shop in two or three friendly sentences. The "
        "context is the customer's name. If you do not know a fact, say a teammate will confirm it.",
        prompt = "{{prompt}}",
    )
    b.e("message.value", "reply.prompt")
    b.e("name.value", "reply.context")
    b.n("draft", "output.result", 3, 1, name = "reply")
    b.e("reply.text", "draft.value")
    b.note(
        "tip",
        "Approve the webhook and switch it on to get its secret URL. A POST answers with the run id at once; "
        "Run by hand uses the test payload.",
        0,
        2,
        1.8,
        0.7,
    )
    return b


def _image_model(b: B, ref: str, choice: dict, note: str) -> None:
    b.doc["requirements"]["models"].append(
        {"ref": ref, "kind": "image", "note": note, "optional": False, **choice}
    )


def lora_style() -> B:
    b = B(
        "lora-style",
        "LoRA Style Compare",
        "pipeline",
        "Draw one prompt with and without a style LoRA, side by side with labels.",
        tags = ["image", "lora", "compare"],
        seed = 31,
    )
    _image_model(b, "krea", KREA, "Krea 2 Turbo; its curated style LoRAs download on first use.")
    b.n("what", "input.text", 0, 1, value = "a lighthouse on a cliff at dawn, gulls overhead")
    b.n("plain", "image.generate", 1, 0.5, label = "No LoRA", model = KREA, size = "1024x1024", seed = 5)
    b.n(
        "styled",
        "image.generate",
        1,
        1.6,
        label = "With LoRA",
        model = KREA,
        size = "1024x1024",
        seed = 5,
        loras = [{"id": "krea/Krea-2-LoRA-softwatercolor", "weight": 1.0}],
        style = "Art Deco watercolor style",
    )
    b.e("what.text", "plain.prompt")
    b.e("what.text", "styled.prompt")
    b.n("both", "logic.collect", 2, 1.1)
    b.e("plain.image", "both.items")
    b.e("styled.image", "both.items")
    b.n(
        "sheet",
        "image.grid",
        3,
        1.1,
        columns = 2,
        caption_style = "none",
        cell_size = 512,
        column_labels = ["No LoRA", "Soft Watercolor LoRA"],
    )
    b.e("both.list", "sheet.images")
    b.n("show", "output.preview", 4, 1.1)
    b.e("sheet.image", "show.value")
    return b


def hires_fix() -> B:
    b = B(
        "hires-fix",
        "Hires Fix",
        "pipeline",
        "Render small, upscale 2x and refine the details in the same request (A1111 / Forge hires fix).",
        tags = ["image", "hires", "upscale"],
        seed = 32,
    )
    # Hires fix needs an image-to-image path, which FLUX.2-klein does not have in Studio yet.
    _image_model(
        b, "zimage", ZIMAGE, "Z-Image Turbo; renders the base image and refines the 2x upscale."
    )
    b.n(
        "what",
        "input.text",
        0,
        1,
        value = "a cozy reading nook with plants and warm lamplight, detailed",
    )
    b.n(
        "draw",
        "image.generate",
        1,
        1,
        model = ZIMAGE,
        size = "768x768",
        hires_scale = 2.0,
        hires_denoise = 0.35,
        seed = 9,
    )
    b.e("what.text", "draw.prompt")
    b.n("show", "output.preview", 2, 1)
    b.e("draw.image", "show.value")
    return b


def face_fix() -> B:
    b = B(
        "face-fix",
        "Face Fix",
        "pipeline",
        "Find the face with a text mask, then repaint only that area at full resolution (a FaceDetailer "
        "built from two blocks).",
        tags = ["image", "mask", "inpaint"],
        seed = 33,
    ).need("image")
    b.n(
        "what",
        "input.text",
        0,
        1,
        value = "full body photo of a hiker on a mountain trail, small in frame",
    )
    b.n("draw", "image.generate", 1, 1, label = "Picture", model = IMAGE, size = "1024x1024", seed = 4)
    b.e("what.text", "draw.prompt")
    b.n("face", "image.segment", 2, 0.4, subject = "the face", grow = 12)
    b.e("draw.image", "face.image")
    b.n(
        "fix",
        "image.generate",
        3,
        1,
        label = "Repaint face",
        model = IMAGE,
        prompt = "a detailed natural face, sharp eyes, soft light",
        # Repaint the whole picture with the face mask: the masked-crop path pastes the crop back oversized on FLUX.2-klein.
        strength = 0.6,
        seed = 4,
    )
    b.e("draw.image", "fix.init_image")
    b.e("face.mask", "fix.mask")
    b.n("both", "logic.collect", 4, 0.4)
    b.e("draw.image", "both.items")
    b.e("fix.image", "both.items")
    b.n(
        "sheet",
        "image.grid",
        5,
        1,
        columns = 2,
        caption_style = "none",
        cell_size = 512,
        column_labels = ["Before", "After"],
    )
    b.e("both.list", "sheet.images")
    b.n("show", "output.preview", 6, 1)
    b.e("sheet.image", "show.value")
    return b


def outpaint() -> B:
    b = B(
        "outpaint",
        "Outpaint",
        "pipeline",
        "Turn a square picture into a wide one: extend the canvas and let the model paint the new sides.",
        tags = ["image", "mask", "outpaint"],
        seed = 34,
    ).need("image")
    b.n("what", "input.text", 0, 1, value = "a quiet harbour town at sunset, boats on calm water")
    b.n("draw", "image.generate", 1, 1, label = "Square", model = IMAGE, size = "768x768", seed = 2)
    b.e("what.text", "draw.prompt")
    b.n("wide", "image.pad", 2, 1, left = 192, right = 192, top = 0, bottom = 0, feather = 24)
    b.e("draw.image", "wide.image")
    b.n("fill", "image.generate", 3, 1, label = "Paint the sides", model = IMAGE, strength = 1.0, seed = 2)
    b.e("what.text", "fill.prompt")
    b.e("wide.image", "fill.init_image")
    b.e("wide.mask", "fill.mask")
    b.n("show", "output.preview", 4, 1)
    b.e("fill.image", "show.value")
    return b


def controlnet_canny() -> B:
    b = B(
        "controlnet-canny",
        "ControlNet Sketch to Picture",
        "pipeline",
        "Keep the outlines of your photo or sketch (canny edges) and draw something new inside them.",
        tags = ["image", "controlnet"],
        seed = 35,
    )
    _image_model(
        b, "flux1", FLUX1, "FLUX.1 dev; the FLUX.1 Union Pro ControlNet downloads on first use."
    )
    b.n("photo", "input.image", 0, 0.5, label = "Your photo or sketch")
    b.n("what", "input.text", 0, 1.6, value = "a watercolor painting of the same scene in autumn")
    b.n(
        "draw",
        "image.generate",
        1,
        1,
        model = FLUX1,
        size = "1024x1024",
        controlnet = "flux-union-pro",
        control_type = "canny",
        control_strength = 0.7,
        control_end = 0.8,
        seed = 8,
    )
    b.e("photo.image", "draw.control_image")
    b.e("what.text", "draw.prompt")
    b.n("show", "output.preview", 2, 1)
    b.e("draw.image", "show.value")
    b.note(
        "tip", "Upload a photo or a line drawing in the first block, then Run.", 0, 2.6, 1.6, 0.6
    )
    return b


def xy_plot() -> B:
    b = B(
        "xy-plot",
        "XY Plot: Style x Seed",
        "pipeline",
        "Draw every style with every seed and lay the results out as a labelled grid, to pick a look "
        "before a final render.",
        tags = ["image", "xy plot", "compare", "for each"],
        seed = 36,
    ).need("image")
    styles = ["watercolor", "pixel art", "film photo"]
    seeds = [1, 2, 3]
    b.n("grid", "logic.product", 0, 1, label = "Combinations", axes = {"style": styles, "seed": seeds})
    b.n("each", "logic.for_each", 1, 1, group = "xy_loop", max_items = 9)
    b.e("grid.items", "each.items")
    b.n(
        "prompt",
        "text.template",
        2,
        0.6,
        group = "xy_loop",
        template = "a red fox in a snowy forest, {{style}} style",
    )
    b.n("seed", "json.get", 2, 1.6, group = "xy_loop", path = "seed")
    b.e("each.item", "prompt.values")
    b.e("each.item", "seed.json")
    b.n("paint", "image.generate", 3, 1, group = "xy_loop", model = IMAGE, size = "512x512")
    b.e("prompt.text", "paint.prompt")
    b.e("seed.value", "paint.seed")
    b.e("paint.image", "each.collect")
    b.group("xy_loop", "For each combination", (1, 0.2, 3, 2.2), "#06b6d4")
    b.n(
        "sheet",
        "image.grid",
        4.4,
        1,
        columns = len(seeds),
        caption_style = "none",
        cell_size = 384,
        row_labels = styles,
        column_labels = [f"seed {s}" for s in seeds],
    )
    b.e("each.results", "sheet.images")
    b.n("show", "output.preview", 5.4, 1)
    b.e("sheet.image", "show.value")
    b.note(
        "tip",
        "Edit the lists in Combinations to sweep other styles or seeds; keep the grid labels in step.",
        0,
        2.2,
        1.6,
        0.6,
    )
    return b


REVIEW_TOPICS = ["price", "quality", "delivery", "support", "ease of use"]


def review_insights() -> B:
    b = B(
        "review-insights",
        "Review Insights",
        "pipeline",
        "Reads customer reviews one by one, tags each with a topic and sentiment, then counts them into a "
        "table and a chart with a short summary of what to fix first.",
        tags = ["table", "chart", "extract", "for each"],
        seed = 54,
    ).need("chat")
    b.n(
        "reviews",
        "input.list",
        0,
        1,
        label = "Reviews",
        items = [
            "Arrived a day early and the box was well packed. Great start.",
            "The handle snapped after two weeks. Feels cheap for the money.",
            "Support answered my question in five minutes and sorted the refund.",
            "Way too expensive compared with the shop down the road.",
            "Setup took one minute, my dad figured it out on his own.",
            "Courier left it in the rain and the box was soaked through.",
            "Solid build, heavy and well finished. Worth every penny.",
            "Emailed support three times and never got a reply.",
        ],
    )
    b.n("each", "logic.for_each", 1, 1, group = "r_loop", max_items = 16)
    b.e("reviews.items", "each.items")
    b.n(
        "tag",
        "llm.extract",
        2,
        1,
        group = "r_loop",
        label = "Topic and sentiment",
        model = CHAT,
        instructions = "Pick the one topic the review is mostly about. stars is the rating the writer would "
        "most likely give, 1 to 5.",
        json_schema = {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "enum": REVIEW_TOPICS},
                "sentiment": {"type": "string", "enum": ["positive", "negative", "mixed"]},
                "stars": {"type": "integer", "minimum": 1, "maximum": 5},
                "review": {"type": "string"},
            },
            "required": ["topic", "sentiment", "stars", "review"],
        },
    )
    b.e("each.item", "tag.text")
    b.e("tag.json", "each.collect")
    b.group("r_loop", "For each review", (1, 0.6, 2, 1.6), "#0d9488")
    b.n(
        "cols",
        "json.set",
        3.2,
        1,
        label = "Pick columns",
        keep_only = True,
        fields = {k: {"var": f"item.{k}"} for k in ("topic", "sentiment", "stars", "review")},
    )
    b.e("each.results", "cols.value")
    b.n("rows", "table.concat", 4.2, 1, label = "Review table")
    b.e("cols.value", "rows.tables")
    b.n(
        "mix",
        "table.aggregate",
        5.2,
        1.2,
        label = "Count by topic and sentiment",
        group_by = ["topic", "sentiment"],
        metrics = ["stars"],
        stat = "mean",
    )
    b.e("rows.table", "mix.table")
    b.n(
        "plot",
        "table.chart",
        6.2,
        1.2,
        label = "Mentions chart",
        type = "bar",
        x = "topic",
        y = ["n"],
    )
    b.doc["nodes"][-1]["params"]["group"] = "sentiment"
    b.e("mix.table", "plot.table")
    b.n(
        "scores",
        "table.aggregate",
        5.2,
        0.2,
        label = "Average stars per topic",
        group_by = ["topic"],
        metrics = ["stars"],
        stat = "mean",
    )
    b.e("rows.table", "scores.table")
    b.n(
        "summary",
        "llm.generate",
        5.2,
        2.4,
        label = "What to fix first",
        model = CHAT,
        temperature = 0.3,
        max_tokens = 300,
        system = "You are a product manager. In three short sentences say what customers like, what hurts "
        "most, and the one thing to fix first. Use only the tagged reviews in the context.",
        prompt = "Summarise these tagged reviews.",
    )
    b.e("cols.value", "summary.context")
    b.n("show_chart", "output.preview", 7.2, 1.2)
    b.e("plot.chart", "show_chart.value")
    b.n("table_out", "output.result", 6.2, 0.2, name = "scores")
    b.e("scores.table", "table_out.value")
    b.n("summary_out", "output.result", 6.2, 2.4, name = "summary")
    b.e("summary.text", "summary_out.value")
    b.n("all_out", "output.result", 4.2, 2, name = "reviews")
    b.e("rows.table", "all_out.value")
    b.note(
        "tip",
        "Paste your own reviews into the list, one per line, and edit the topics in Topic and sentiment.",
        0,
        2,
        1.6,
        0.6,
    )
    return b


def image_alt_text() -> B:
    b = B(
        "image-alt-text",
        "Image Alt Text",
        "pipeline",
        "Writes accessibility text for a picture: a short alt text, a longer description and any words "
        "visible in the image.",
        tags = ["vision", "accessibility", "json"],
        seed = 55,
    ).need("vision")
    b.n("photo", "input.image", 0, 1, label = "Your picture")
    b.n(
        "look",
        "llm.vision",
        1,
        1,
        label = "Describe for screen readers",
        model = VISION,
        question = "Write accessibility text for this image. alt_text: one sentence under 125 characters "
        "that says what matters, without starting with 'image of'. description: two or three sentences "
        "for someone who cannot see it, covering people, setting, colours and mood. visible_text: every "
        "word printed in the image, or an empty string.",
        json_schema = {
            "type": "object",
            "properties": {
                "alt_text": {"type": "string"},
                "description": {"type": "string"},
                "visible_text": {"type": "string"},
            },
            "required": ["alt_text", "description", "visible_text"],
        },
        max_tokens = 600,
    )
    b.e("photo.image", "look.image")
    for row, (field, name) in enumerate(
        (("alt_text", "alt"), ("description", "long"), ("visible_text", "words"))
    ):
        b.n(f"get_{name}", "json.get", 2, row * 0.9 + 0.1, path = field)
        b.e("look.json", f"get_{name}.json")
        b.n(f"{name}_out", "output.result", 3, row * 0.9 + 0.1, name = field)
        b.e(f"get_{name}.value", f"{name}_out.value")
    b.n("show", "output.preview", 1, 2.2, label = "Picture and alt text")
    b.e("photo.image", "show.value")
    b.e("get_alt.value", "show.value")
    b.note(
        "tip",
        "Upload a photo, chart or screenshot in the first block, then Run. Add an <img alt> straight from "
        "alt_text.",
        0,
        2.3,
        0.9,
        0.8,
    )
    return b


def background_swap() -> B:
    b = B(
        "background-swap",
        "Background Swap",
        "pipeline",
        "Cuts the product out of your photo and places it on a freshly generated scene, with a before and "
        "after sheet.",
        tags = ["image", "product", "remove background", "composite"],
        seed = 56,
    ).need("image")
    b.n("photo", "input.image", 0, 0.5, label = "Product photo")
    b.n(
        "cut",
        "image.remove_bg",
        1,
        0.5,
        label = "Cut out product",
        subject = "the product",
        trim = True,
    )
    b.e("photo.image", "cut.image")
    b.n(
        "scene",
        "input.text",
        0,
        1.6,
        label = "New background",
        value = "an empty white marble kitchen counter by a sunny window, soft morning light, plants blurred "
        "in the background, product photography",
    )
    b.n(
        "backdrop",
        "image.generate",
        1,
        1.6,
        label = "Paint the background",
        model = IMAGE,
        size = "1024x1024",
        seed = 12,
    )
    b.e("scene.text", "backdrop.prompt")
    b.n(
        "place",
        "image.composite",
        2,
        1,
        label = "Place product",
        x = 0.5,
        y = 0.62,
        scale = 0.32,
    )
    b.e("backdrop.image", "place.image")
    b.e("cut.image", "place.overlay")
    b.n("steps", "logic.collect", 3, 0.4)
    b.e("photo.image", "steps.items")
    b.e("cut.image", "steps.items")
    b.e("place.image", "steps.items")
    # A list, not a grid: grid cells crop square, which would cut into the trimmed cut-out.
    b.n("show", "output.preview", 4, 0.4, label = "Before and after")
    b.e("steps.list", "show.value")
    b.n("save", "output.library", 3, 1.6, folder = "Product shots")
    b.e("place.image", "save.value")
    b.note(
        "tip",
        "Upload a product photo, describe the scene, then Run. Move or resize the product with X, Y and "
        "Width on Place product.",
        0,
        2.6,
        1.8,
        0.6,
    )
    return b


def document_audiobook() -> B:
    b = B(
        "document-audiobook",
        "Document Audiobook",
        "pipeline",
        "Load a PDF, Word or text file, cut it into short passages, read each one aloud and join them into one "
        "audiobook in your Library.",
        tags = ["documents", "audio", "tts", "for each"],
        seed = 2028,
    ).need("tts")
    b.n("doc", "input.file", 0, 1, label = "Your document")
    b.n("read", "doc.load", 1, 1)
    b.e("doc.file", "read.file")
    b.n(
        "parts",
        "text.split",
        2,
        1,
        label = "Passages",
        mode = "chunks",
        unit = "words",
        chunk_size = 120,
        overlap = 0,
    )
    b.e("read.text", "parts.text")
    b.n("each", "logic.for_each", 3, 1, group = "read_loop", max_items = 8)
    b.n(
        "speak",
        "audio.speak",
        4,
        1,
        group = "read_loop",
        label = "Read aloud",
        model = TTS,
        voice = "af_heart",
    )
    b.group("read_loop", "For each passage", (3, 0.4, 2.2, 1.6), "#10b981")
    b.e("parts.items", "each.items")
    b.e("each.item", "speak.text")
    b.e("speak.audio", "each.collect")
    b.n("join", "audio.concat", 5.4, 1, gap_ms = 400)
    b.e("each.results", "join.clips")
    b.n("listen", "output.preview", 6.4, 0.5)
    b.e("join.audio", "listen.value")
    b.n("save", "output.library", 6.4, 1.5, folder = "Audiobooks")
    b.e("join.audio", "save.value")
    b.n("text", "output.result", 3, 2.6, name = "passages")
    b.e("parts.items", "text.value")
    b.note(
        "tip",
        "Upload a document in the first block, then Run. It reads the first 8 passages (about 1,000 words); "
        "raise Max items in For Each for longer books.",
        0,
        2.2,
        2,
        0.7,
    )
    return b


def resume_tailor() -> B:
    b = B(
        "resume-tailor",
        "Resume Tailor",
        "pipeline",
        "Upload your resume and paste a job post: a local model maps what you already match, lists the gaps, "
        "then writes tailored resume bullets and a short cover letter from your real experience.",
        tags = ["documents", "llm", "json", "career"],
        seed = 2029,
    ).need("chat")
    b.n("resume", "input.file", 0, 0.5, label = "Your resume")
    b.n("read", "doc.load", 1, 0.5)
    b.e("resume.file", "read.file")
    b.n(
        "job",
        "input.text",
        0,
        1.6,
        label = "Job post",
        value = "Data Analyst, Riverside Health. We need someone who writes SQL every day, builds dashboards in "
        "Tableau or Power BI, cleans messy data in Python (pandas), and explains findings to non-technical "
        "teams. Nice to have: A/B testing, healthcare data, dbt.",
    )
    b.n(
        "fit",
        "llm.generate",
        2,
        1,
        label = "Match to the job",
        model = CHAT,
        temperature = 0.2,
        max_tokens = 1200,
        system = "You are a careful career coach. Compare the resume in the context with the job post. A match "
        "needs evidence quoted or paraphrased from the resume; anything the job asks for that the resume does "
        "not show is a gap. Never invent experience.",
        prompt = "Job post:\n{{prompt}}",
        json_schema = RESUME_FIT,
    )
    b.e("job.text", "fit.prompt")
    b.e("read.text", "fit.context")
    b.n("gaps", "json.get", 3, 1.9, path = "gaps")
    b.e("fit.json", "gaps.json")
    b.n("missing", "output.result", 4, 1.9, name = "missing skills")
    b.e("gaps.value", "missing.value")
    b.n(
        "write",
        "llm.generate",
        3,
        0.6,
        label = "Write for this job",
        model = CHAT,
        temperature = 0.5,
        max_tokens = 1500,
        system = "You rewrite resumes. Use only facts from the resume and the match notes in the context; "
        "reuse the job's keywords where they are true. Bullets start with a strong verb and stay under 25 words. "
        "The cover letter is three short paragraphs and does not claim the gaps.",
        prompt = "Job post:\n{{prompt}}\n\nWrite 5 to 7 tailored resume bullets and a cover letter.",
        json_schema = {
            "type": "object",
            "properties": {
                "bullets": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 8,
                    "items": {"type": "string"},
                },
                "cover_letter": {"type": "string"},
            },
            "required": ["bullets", "cover_letter"],
        },
    )
    b.e("job.text", "write.prompt")
    b.e("read.text", "write.context")
    b.e("fit.json", "write.context")
    b.n("bullets", "json.get", 4, 0.2, path = "bullets")
    b.e("write.json", "bullets.json")
    b.n("letter", "json.get", 4, 1, path = "cover_letter")
    b.e("write.json", "letter.json")
    b.n("show_bullets", "output.result", 5, 0.2, name = "tailored bullets")
    b.e("bullets.value", "show_bullets.value")
    b.n("show_letter", "output.result", 5, 1, name = "cover letter")
    b.e("letter.value", "show_letter.value")
    b.note(
        "tip",
        "Upload your resume (PDF, Word or text) in the first block and paste the job post, then Run.",
        0,
        2.6,
        1.8,
        0.6,
    )
    return b


def study_cards() -> B:
    b = B(
        "study-cards",
        "Study Cards",
        "pipeline",
        "Paste your notes and get a deck of quiz questions with answers, duplicates removed, ready to study.",
        tags = ["learning", "llm", "json", "list"],
        seed = 2030,
    ).need("chat")
    b.n(
        "notes",
        "input.text",
        0,
        1,
        label = "Your notes",
        value = "Photosynthesis turns light energy into chemical energy in chloroplasts. The light reactions "
        "happen in the thylakoid membranes: chlorophyll absorbs light, water is split and oxygen is released, "
        "and the energy is stored in ATP and NADPH. The Calvin cycle runs in the stroma: the enzyme RuBisCO "
        "fixes carbon dioxide onto RuBP, and ATP and NADPH from the light reactions turn the product into the "
        "sugar G3P. Plants use G3P to build glucose, sucrose and starch. Light, carbon dioxide and temperature "
        "all limit the rate of photosynthesis.",
    )
    b.n(
        "write",
        "llm.generate",
        1,
        1,
        label = "Write cards",
        model = CHAT,
        temperature = 0.3,
        max_tokens = 1500,
        system = "You write flashcards for students. Each card tests one fact from the notes; the question is "
        "short and specific, the answer is one sentence. Never add facts that are not in the notes.",
        prompt = "Make 8 to 12 flashcards from these notes:\n\n{{prompt}}",
        json_schema = {
            "type": "object",
            "properties": {
                "cards": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                        },
                        "required": ["question", "answer"],
                    },
                }
            },
            "required": ["cards"],
        },
    )
    b.e("notes.text", "write.prompt")
    b.n("cards", "json.get", 2, 1, path = "cards")
    b.e("write.json", "cards.json")
    b.n(
        "unique",
        "list.dedupe",
        3,
        1,
        label = "One per question",
        key = {"op": "lower", "args": [{"var": "item.question"}]},
    )
    b.e("cards.value", "unique.items")
    b.n("deck", "output.result", 4, 0.3, name = "cards")
    b.e("unique.items", "deck.value")
    b.n("removed", "output.result", 4, 1.9, name = "duplicates removed")
    b.e("unique.removed", "removed.value")
    b.n(
        "as_text",
        "list.map",
        4,
        1.1,
        label = "Question and answer",
        expression = {
            "op": "concat",
            "args": ["Q: ", {"var": "item.question"}, "\nA: ", {"var": "item.answer"}],
        },
    )
    b.e("unique.items", "as_text.items")
    b.n("sheet", "list.join", 5, 1.1, separator = "\\n\\n")
    b.e("as_text.items", "sheet.items")
    b.n("show", "output.preview", 6, 1.1)
    b.e("sheet.text", "show.value")
    b.note("tip", "Paste your own notes in the first block, then Run.", 0, 2.2, 1.6, 0.5)
    return b


def release_notes() -> B:
    b = B(
        "release-notes",
        "Release Notes from Changes",
        "pipeline",
        "Paste the merged changes: a local model sorts each one into features, fixes and breaking changes, drops "
        "internal chores and writes release notes, with the breaking changes listed on their own.",
        tags = ["text", "for each", "classify", "automation"],
        seed = 2031,
    ).need("chat")
    b.n("version", "input.text", 0, 0.4, label = "Version", value = "2.4.0")
    b.n(
        "changes",
        "input.list",
        0,
        1.4,
        label = "Merged changes",
        items = [
            "Add dark mode to the settings page",
            "Fix crash when exporting an empty playlist",
            "Rename the --out flag to --output (old flag removed)",
            "Bump the CI runner image",
            "Speed up search by caching the index between queries",
            "Fix the wrong total on invoices with discounts",
            "Drop support for Python 3.8",
            "Document the new webhook settings",
        ],
    )
    b.n("each", "logic.for_each", 1, 1.4, group = "sort_loop", max_items = 32)
    b.e("changes.items", "each.items")
    b.n(
        "sort",
        "llm.extract",
        2,
        1.4,
        group = "sort_loop",
        label = "Sort the change",
        model = CHAT,
        instructions = "note is one short past-tense sentence a user understands. kind is one of: feature "
        "(something new users can do or that got faster), fix (a bug that was fixed), breaking (users must change "
        "their code, flags or setup to upgrade, such as a renamed or removed option or a dropped platform), docs "
        "(documentation only), internal (CI, build, tooling and refactors users never notice). A fix or a "
        "speed-up is never breaking.",
        # The note comes first, so the model has described the change before it picks a kind.
        json_schema = {
            "type": "object",
            "properties": {
                "note": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["feature", "fix", "breaking", "docs", "internal"],
                },
            },
            "required": ["note", "kind"],
        },
    )
    b.e("each.item", "sort.text")
    b.e("sort.json", "each.collect")
    b.group("sort_loop", "For each change", (1, 0.9, 2, 1.6), "#06b6d4")
    b.n(
        "for_users",
        "list.filter",
        3.2,
        1,
        label = "Users notice it",
        condition = {"op": "!=", "args": [{"var": "item.kind"}, "internal"]},
    )
    b.e("each.results", "for_users.items")
    b.n("left_out", "output.preview", 4.2, 2.2)
    b.e("for_users.dropped", "left_out.value")
    b.n("ask", "text.template", 3.2, 0, template = "Write the release notes for version {{values}}.")
    b.e("version.text", "ask.values")
    b.n(
        "write",
        "llm.generate",
        4.2,
        0.6,
        label = "Write the notes",
        model = CHAT,
        temperature = 0.3,
        max_tokens = 900,
        system = "You write release notes in Markdown from the sorted changes in the context: a title with the "
        "version, then Breaking changes, New features, Fixes and Docs sections (skip empty ones), one bullet per "
        "change. Add a one-line upgrade hint under each breaking change.",
        prompt = "{{prompt}}",
    )
    b.e("ask.text", "write.prompt")
    b.e("for_users.items", "write.context")
    b.n("notes", "output.result", 5.2, 0.6, name = "release_notes")
    b.e("write.text", "notes.value")
    b.n(
        "breaking",
        "list.filter",
        4.2,
        1.6,
        label = "Breaking only",
        condition = {"op": "==", "args": [{"var": "item.kind"}, "breaking"]},
    )
    b.e("for_users.items", "breaking.items")
    b.n("lines", "list.map", 5.2, 1.6, label = "Their notes", expression = {"var": "item.note"})
    b.e("breaking.items", "lines.items")
    b.n("must_read", "output.result", 6.2, 1.6, name = "breaking")
    b.e("lines.items", "must_read.value")
    b.note(
        "tip",
        "Paste a changelog or merged PR titles, one per line. Change the kinds in Sort the change to match "
        "your project's labels.",
        0,
        2.4,
        1.8,
        0.7,
    )
    return b


def localization_kit() -> B:
    b = B(
        "localization-kit",
        "Localization Kit",
        "pipeline",
        "Translate an app's UI strings into several languages at once with a local model, keeping every "
        "{placeholder} and key unchanged.",
        tags = ["text", "translate", "for each", "json"],
        seed = 2032,
    ).need("chat")
    b.n(
        "strings",
        "input.param",
        0,
        0.5,
        label = "UI strings",
        name = "strings",
        type = "json",
        default = json.dumps(LOCALIZATION_STRINGS, indent = 2),
    )
    b.n(
        "languages",
        "input.list",
        0,
        1.5,
        label = "Languages",
        items = ["Spanish", "German", "Japanese", "Brazilian Portuguese"],
    )
    b.n("each", "logic.for_each", 1, 1.5, group = "lang_loop", max_items = 16)
    b.e("languages.items", "each.items")
    b.n(
        "ask",
        "text.template",
        2,
        1.5,
        group = "lang_loop",
        template = "Target language: {{values}}. Translate every UI string in the context into {{values}}.",
    )
    b.e("each.item", "ask.values")
    b.n(
        "translate",
        "llm.generate",
        3,
        1.5,
        group = "lang_loop",
        model = CHAT,
        temperature = 0.2,
        max_tokens = 1500,
        system = "You are a professional software translator. The context is a JSON object of UI strings by key. "
        "Translate each value into the language the user names and return them under the same keys. Keep "
        "placeholders in curly braces such as {name} exactly as written, untranslated. Match the length and tone "
        "of a button or message.",
        prompt = "{{prompt}}",
        # Strings come first: a small model asked for the language name first tends to write a translation there.
        json_schema = {
            "type": "object",
            "properties": {
                "strings": {
                    "type": "object",
                    "description": "Every key from the context, mapped to its translation",
                    "additionalProperties": {"type": "string"},
                },
                "language": {"type": "string", "description": "The target language"},
            },
            "required": ["strings", "language"],
        },
    )
    b.e("ask.text", "translate.prompt")
    b.e("strings.value", "translate.context")
    b.e("translate.json", "each.collect")
    b.group("lang_loop", "For each language", (1, 1, 3, 1.6), "#06b6d4")
    b.n("kit", "output.result", 4.4, 1.5, name = "translations")
    b.e("each.results", "kit.value")
    b.note(
        "tip",
        "Paste your own strings as JSON (key: text) into UI strings, or fill it from the Run dialog or an MCP "
        "agent. Add languages one per line.",
        0,
        2.5,
        1.8,
        0.7,
    )
    return b


def _or_zero(field: str) -> dict:
    return {"op": "if", "args": [{"var": f"item.{field}"}, {"var": f"item.{field}"}, 0]}


def _missing(field: str) -> dict:
    return {"op": "==", "args": [{"var": f"item.{field}"}, None]}


def inventory_reconcile() -> B:
    b = B(
        "inventory-reconcile",
        "Inventory Reconcile",
        "pipeline",
        "Match a stock count against the system's records by SKU and list every discrepancy, biggest cost "
        "first. Plain data blocks, no model: it runs in a blink.",
        tags = ["data", "json", "transform", "table"],
        seed = 2033,
    )
    b.n(
        "stock",
        "input.param",
        0,
        0.5,
        label = "System stock",
        name = "stock",
        type = "json",
        default = json.dumps(INVENTORY_STOCK, indent = 2),
    )
    b.n(
        "count",
        "input.param",
        0,
        1.5,
        label = "Shelf count",
        name = "count",
        type = "json",
        default = json.dumps(INVENTORY_COUNT, indent = 2),
    )
    b.n("join", "json.merge", 1, 1, label = "Join on SKU", mode = "by_key", key = "sku")
    b.e("stock.value", "join.a")
    b.e("count.value", "join.b")
    b.n(
        "difference",
        "json.set",
        2,
        1,
        label = "Difference",
        fields = {"difference": {"op": "-", "args": [_or_zero("counted"), _or_zero("on_hand")]}},
    )
    b.e("join.value", "difference.value")
    b.n(
        "status",
        "json.set",
        3,
        1,
        label = "Status and cost",
        fields = {
            "status": {
                "op": "if",
                "args": [
                    _missing("on_hand"),
                    "not in system",
                    {
                        "op": "if",
                        "args": [
                            _missing("counted"),
                            "not counted",
                            {
                                "op": "if",
                                "args": [
                                    {"op": "<", "args": [{"var": "item.difference"}, 0]},
                                    "short",
                                    {
                                        "op": "if",
                                        "args": [
                                            {"op": ">", "args": [{"var": "item.difference"}, 0]},
                                            "over",
                                            "ok",
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
            "value": {
                "op": "round",
                "args": [
                    {"op": "*", "args": [{"var": "item.difference"}, _or_zero("unit_cost")]},
                    2,
                ],
            },
        },
    )
    b.e("difference.value", "status.value")
    b.n(
        "off",
        "list.filter",
        4,
        0.6,
        label = "Discrepancies",
        condition = {"op": "!=", "args": [{"var": "item.status"}, "ok"]},
    )
    b.e("status.value", "off.items")
    b.n(
        "worst",
        "list.sort",
        5,
        0.6,
        label = "Biggest cost first",
        key = {"op": "abs", "args": [{"var": "item.value"}]},
        descending = True,
    )
    b.e("off.items", "worst.items")
    b.n("report", "output.result", 6, 0.6, name = "discrepancies")
    b.e("worst.items", "report.value")
    b.n(
        "totals",
        "table.aggregate",
        4,
        1.7,
        label = "Totals by status",
        group_by = ["status"],
        metrics = ["difference", "value"],
        stat = "sum",
    )
    b.e("status.value", "totals.table")
    b.n("summary", "output.preview", 5, 1.7)
    b.e("totals.table", "summary.value")
    b.note(
        "tip",
        "Paste exports from your shop and your counting sheet as JSON lists with a sku field. Rows only on one "
        "side show up as not counted or not in system.",
        0,
        2.5,
        1.8,
        0.7,
    )
    return b


def campaign_kit() -> B:
    b = B(
        "campaign-kit",
        "Ad Campaign Kit",
        "pipeline",
        "Describe a product: a local model writes the headline and an image brief, one image is generated, then "
        "cropped into square, portrait and landscape ads with the headline on each.",
        tags = ["image", "marketing", "text", "resize"],
        seed = 2034,
    ).need("chat", "image")
    b.n(
        "brief",
        "input.text",
        0,
        1,
        label = "Product",
        value = "Moss & Mist, a small-batch cold brew coffee in a frosted glass bottle, brewed for 20 hours, "
        "for people who like slow mornings.",
    )
    b.n(
        "copy",
        "llm.generate",
        1,
        1,
        label = "Write the copy",
        model = CHAT,
        temperature = 0.7,
        max_tokens = 400,
        system = "You write ad copy. headline: at most 6 words. tagline: one short sentence. image_prompt: a "
        "photo brief for an image model with the product as the clear subject in the middle, soft natural light, "
        "plenty of empty space above it for text, no words or logos in the image.",
        prompt = "{{prompt}}",
        json_schema = {
            "type": "object",
            "properties": {
                "headline": {"type": "string"},
                "tagline": {"type": "string"},
                "image_prompt": {"type": "string"},
            },
            "required": ["headline", "tagline", "image_prompt"],
        },
    )
    b.e("brief.text", "copy.prompt")
    b.n("headline", "json.get", 2, 0.3, path = "headline")
    b.e("copy.json", "headline.json")
    b.n("scene", "json.get", 2, 1.3, label = "Image brief", path = "image_prompt")
    b.e("copy.json", "scene.json")
    b.n("words", "output.result", 2, 2.3, name = "copy")
    b.e("copy.json", "words.value")
    b.n("photo", "image.generate", 3, 1.3, model = IMAGE, size = "1024x1024")
    b.e("scene.value", "photo.prompt")
    formats = [("square", "Square 1:1", 1080, 1080), ("portrait", "Portrait 4:5", 1080, 1350),
               ("landscape", "Landscape 1.91:1", 1200, 628)]  # fmt: skip
    for row, (fid, label, w, h) in enumerate(formats):
        b.n(fid, "image.resize", 4, row, label = label, width = w, height = h, mode = "cover")
        b.e("photo.image", f"{fid}.image")
        b.n(f"{fid}_ad", "image.text", 5, row, label = "Headline", size = 72, position = "top", outline = 3)
        b.e(f"{fid}.image", f"{fid}_ad.image")
        b.e("headline.value", f"{fid}_ad.text")
    b.n("ads", "logic.collect", 6, 1)
    for fid, *_ in formats:
        b.e(f"{fid}_ad.image", "ads.items")
    b.n("show", "output.preview", 7, 0.5, label = "Ad set")
    b.e("ads.list", "show.value")
    b.n("save", "output.library", 7, 1.6, folder = "Campaigns")
    b.e("ads.list", "save.value")
    b.note(
        "tip",
        "One image is generated and reused for every format, so the set stays consistent. Change the sizes in "
        "the Crop / Resize blocks for other placements.",
        0,
        2.2,
        1.8,
        0.7,
    )
    return b


BUILDERS = (
    lantern_picnic,
    sloth_moon,
    infinite_dungeon,
    pocket_village,
    comic_strip,
    podcast,
    storyboard,
    dataset_forge,
    prompt_tournament,
    sticker_pack,
    meme_maker,
    postcard,
    social_clip,
    photo_restyle,
    animate_photo,
    meeting_notes,
    translate_and_dub,
    product_listing,
    song_maker,
    long_form_writer,
    text_sft,
    vision_sft,
    tts_sft,
    embedding_sft,
    model_arena,
    smallville_lite,
    village_rivals,
    remembering_assistant,
    summarise_pdf,
    ask_documents,
    contact_cleanup,
    chat_with_documents,
    support_bot,
    research_agent,
    invoice_extractor,
    email_triage,
    daily_digest,
    webhook_reply,
    lora_style,
    hires_fix,
    face_fix,
    outpaint,
    controlnet_canny,
    xy_plot,
    review_insights,
    image_alt_text,
    background_swap,
    document_audiobook,
    resume_tailor,
    study_cards,
    release_notes,
    localization_kit,
    inventory_reconcile,
    campaign_kit,
    lantern_isle_3d,
    mayor_meadow,
)


def build() -> list[dict]:
    docs = []
    for i, fn in enumerate((*BUILDERS, *_learning_builders())):
        doc = fn().doc
        doc["meta"]["order"] = i
        docs.append(FlowDocument.model_validate(doc).model_dump(mode = "json"))
    return docs


@functools.lru_cache(maxsize = 1)
def _examples() -> dict[str, dict]:
    return {doc["id"]: doc for doc in build()}


def example_ids() -> list[str]:
    return list(_examples())


def example_docs() -> list[dict]:
    """The shared example documents: read only (``example`` returns a copy to edit)."""
    return list(_examples().values())


def example(example_id: str) -> dict | None:
    """A fresh copy of one example document, or None."""
    doc = _examples().get(example_id)
    return copy.deepcopy(doc) if doc is not None else None


def main() -> None:
    for doc in build():
        print(doc["id"], len(doc["nodes"]), "nodes")


if __name__ == "__main__":
    main()
