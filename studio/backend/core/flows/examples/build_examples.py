# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The bundled example flows, built in code so the editor's Examples shelf and the tests read one source.
`python -m core.flows.examples.build_examples` prints each example's id and node count."""

from __future__ import annotations

import copy
import functools
from typing import Any, Optional

from core.flows.schema import FlowDocument

CHAT = {"repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF", "variant": "Q4_K_M"}
VISION = {"repo": "unsloth/Qwen3.5-4B-GGUF", "variant": "Q4_K_M"}
IMAGE = {"repo": "black-forest-labs/FLUX.2-klein-4B", "variant": None}
VIDEO = {"repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "variant": None}
TTS = {"repo": "audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF", "variant": None}

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
    "tts": {
        "ref": "voice",
        "kind": "tts",
        "note": "82M multi-voice TTS (voices af_heart, am_michael, ...).",
        **TTS,
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
            "Narrator: Fog swallowed the path.\nPip: I... can't see my own glow.\nMo: Then I'll carry you. Slowly. Very slowly.",
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
            "Mo: Look! The moon is in the water!\nOttoline: Or is it?",
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
        "Mo: Goodnight, moon. We'll watch you every night.",
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
        style = "comic book panel, bold ink lines, flat bright colors, expressive",
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
        size = "832x480",
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
    b.n(
        "recipe",
        "data.recipe",
        3,
        2.2,
        label = "(Alternative) Data Recipe",
        recipe_id = "",
        rows = 200,
        mode = "full",
    )
    b.doc["nodes"][-1]["disabled"] = True
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
        "Training needs a GPU and stops chat models while it runs. Enable the Data Recipe node (and connect it to "
        "Fine-tune instead of Rows to Dataset) to use one of your saved recipes.",
        3,
        -0.2,
        1.6,
        0.9,
    )
    return b


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
)


def build() -> list[dict]:
    docs = []
    for i, fn in enumerate(BUILDERS):
        doc = fn().doc
        doc["meta"]["order"] = i
        docs.append(FlowDocument.model_validate(doc).model_dump(mode = "json"))
    return docs


@functools.lru_cache(maxsize = 1)
def _examples() -> dict[str, dict]:
    return {doc["id"]: doc for doc in build()}


def example_ids() -> list[str]:
    return list(_examples())


def example(example_id: str) -> dict | None:
    """A fresh copy of one example document, or None."""
    doc = _examples().get(example_id)
    return copy.deepcopy(doc) if doc is not None else None


def main() -> None:
    for doc in build():
        print(doc["id"], len(doc["nodes"]), "nodes")


if __name__ == "__main__":
    main()
