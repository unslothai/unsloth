# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Flow document, the values that travel along edges, the story manifest and the world spec.

This module is the contract shared with the frontend (`studio/frontend/src/features/flows/types.ts`).
A Flow document is pure data: it names node types from the built-in registry and never carries code,
host paths, URLs to fetch, credentials or MCP server configs."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_serializer

FORMAT = "unsloth-flow"
FORMAT_VERSION = 1

MAX_NODES = 2000
MAX_EDGES = 6000
MAX_DOC_BYTES = 8 * 1024 * 1024
MAX_PACKAGE_BYTES = 512 * 1024 * 1024
MAX_ASSET_BYTES = 128 * 1024 * 1024
MAX_ASSETS = 2000
MAX_JSON_DEPTH = 64
MAX_FOREACH_ITEMS = 256
MAX_GPU_OPS_PER_RUN = 2000
MAX_RUN_SECONDS = 24 * 3600
MAX_STRING_PARAM = 200_000

FlowKind = Literal["pipeline", "story", "world"]

PortType = Literal[
    "text",
    "number",
    "bool",
    "json",
    "list",
    "image",
    "audio",
    "video",
    "dataset",
    "model",
    "checkpoint",
    "story",
    "world",
    "any",
    "flow",
]

EdgeKind = Literal["data", "flow", "feedback"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra = "forbid")


class FlowParam(_Strict):
    """A flow-level input exposed in the Run dialog, to MCP `run_flow` and to the player."""

    id: str = Field(pattern = r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
    type: Literal["text", "number", "bool", "json", "image", "audio"] = "text"
    label: Optional[str] = None
    default: Any = None
    description: Optional[str] = None


class ModelRequirement(_Strict):
    ref: str = Field(pattern = r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
    kind: Literal["chat", "vision", "image", "video", "audio", "tts", "stt", "embedding"]
    repo: str = Field(max_length = 300)
    variant: Optional[str] = Field(default = None, max_length = 200)
    optional: bool = False
    note: Optional[str] = None


class AssetRef(_Strict):
    """An embedded file in a package: `assets/<sha256>.<ext>`."""

    id: str = Field(pattern = r"^sha256:[0-9a-f]{64}$")
    mime: str = Field(max_length = 100)
    name: Optional[str] = Field(default = None, max_length = 200)
    bytes: int = Field(ge = 0)


class FlowNode(_Strict):
    id: str = Field(pattern = r"^[A-Za-z0-9_-]{1,64}$")
    type: str = Field(pattern = r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
    v: int = Field(default = 1, ge = 1)
    params: dict[str, Any] = Field(default_factory = dict)
    pos: tuple[float, float] = (0.0, 0.0)
    label: Optional[str] = Field(default = None, max_length = 200)
    group: Optional[str] = Field(default = None, max_length = 64)
    collapsed: bool = False
    disabled: bool = False


class FlowEdge(_Strict):
    id: str = Field(pattern = r"^[A-Za-z0-9_-]{1,96}$")
    source: str
    source_port: str
    target: str
    target_port: str
    kind: EdgeKind = "data"


class FlowGroup(_Strict):
    """A frame around nodes. A "recipe" group is one Data Recipe: its `recipe.column` nodes, the
    recipe-level `config` (seed, model providers and configs, processors), `run` settings and the
    Recipe Studio `ui` hints its seed editor reads. A `recipe.generate` node inside runs it. A "loop"
    group frames a Loop node and the steps it repeats each round."""

    id: str = Field(pattern = r"^[A-Za-z0-9_-]{1,64}$")
    label: str = Field(default = "Group", max_length = 200)
    color: Optional[str] = Field(default = None, pattern = r"^#[0-9a-fA-F]{6}$")
    rect: tuple[float, float, float, float] = (0.0, 0.0, 400.0, 300.0)
    kind: Optional[Literal["recipe", "loop"]] = None
    config: Optional[dict[str, Any]] = None
    run: Optional[dict[str, Any]] = None
    ui: Optional[dict[str, Any]] = None

    @model_serializer(mode = "wrap")
    def _drop_unset_recipe(self, handler):
        # A plain group saves exactly as before.
        data = handler(self)
        for key in ("kind", "config", "run", "ui"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class FlowNote(_Strict):
    id: str = Field(pattern = r"^[A-Za-z0-9_-]{1,64}$")
    text: str = Field(default = "", max_length = 20_000)
    rect: tuple[float, float, float, float] = (0.0, 0.0, 240.0, 120.0)
    color: Optional[str] = Field(default = None, pattern = r"^#[0-9a-fA-F]{6}$")


class FlowDocument(_Strict):
    format: Literal["unsloth-flow"] = FORMAT
    version: int = FORMAT_VERSION
    id: str = Field(pattern = r"^[A-Za-z0-9_-]{1,64}$")
    name: str = Field(min_length = 1, max_length = 200)
    description: str = Field(default = "", max_length = 20_000)
    kind: FlowKind = "pipeline"
    seed: int = Field(default = 0, ge = 0, le = 2**32 - 1)
    params: list[FlowParam] = Field(default_factory = list)
    requirements: dict[str, list[ModelRequirement]] = Field(default_factory = lambda: {"models": []})
    nodes: list[FlowNode] = Field(default_factory = list)
    edges: list[FlowEdge] = Field(default_factory = list)
    groups: list[FlowGroup] = Field(default_factory = list)
    notes: list[FlowNote] = Field(default_factory = list)
    assets: list[AssetRef] = Field(default_factory = list)
    baked: dict[str, dict[str, Any]] = Field(default_factory = dict)
    editor: dict[str, Any] = Field(default_factory = dict)
    meta: dict[str, Any] = Field(default_factory = dict)


class MediaValue(_Strict):
    kind: Literal["image", "audio", "video"]
    ref: str
    url: Optional[str] = None
    mime: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration_s: Optional[float] = None
    seed: Optional[int] = None
    prompt: Optional[str] = None


class DatasetValue(_Strict):
    kind: Literal["dataset"] = "dataset"
    source: Literal["hf", "recipe", "local"]
    ref: str
    rows: Optional[int] = None
    columns: Optional[list[str]] = None


class ModelValue(_Strict):
    kind: Literal["model"] = "model"
    model_kind: str
    repo: str
    variant: Optional[str] = None


class CheckpointValue(_Strict):
    kind: Literal["checkpoint"] = "checkpoint"
    name: str
    base_model: Optional[str] = None


# Expressions: a JSON tree, evaluated by core/flows/expr.py and src/features/flows/expr.ts
# Expr := null | bool | number | string | {"var": "name"} | {"op": OP, "args": [Expr, ...]} | {"list": [Expr...]}
EXPR_OPS = (
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "and",
    "or",
    "not",
    "+",
    "-",
    "*",
    "/",
    "%",
    "min",
    "max",
    "abs",
    "round",
    "len",
    "contains",
    "concat",
    "if",
    "lower",
    "upper",
    "rand",
    "randint",
    "dice",
)
Expr = Any


class StoryCharacter(_Strict):
    id: str
    name: str
    color: Optional[str] = None
    look: Optional[str] = None
    voice: Optional[str] = None
    portrait: Optional[dict[str, Any]] = None
    expressions: dict[str, dict[str, Any]] = Field(default_factory = dict)
    side: Literal["left", "right", "center"] = "left"


StoryTransition = Literal["dissolve", "fade", "flash", "slideleft", "slideright", "none"]
StoryPosition = Literal["left", "center", "right", "twoleft", "tworight"]


class StoryStage(_Strict):
    """Ren'Py-style directions applied when the reader reaches the line; a line with empty text is only these."""

    show: Optional[str] = None
    hide: Optional[str] = None
    at: Optional[StoryPosition] = None
    expression: Optional[str] = None
    with_: Optional[StoryTransition] = Field(default = None, alias = "with")
    sound: Optional[str] = None
    pause: Optional[float] = Field(default = None, ge = 0, le = 30)

    model_config = ConfigDict(extra = "forbid", populate_by_name = True)


class StoryLine(_Strict):
    speaker: Optional[str] = None  # character id, None = narrator
    text: str
    expression: Optional[str] = None
    cond: Expr = Field(default = None, alias = "if")
    voice: Optional[dict[str, Any]] = None
    stage: Optional[StoryStage] = None

    model_config = ConfigDict(extra = "forbid", populate_by_name = True)


class StoryChoice(_Strict):
    id: str
    text: str
    goto: str
    cond: Expr = Field(default = None, alias = "if")
    set: dict[str, Expr] = Field(default_factory = dict)

    model_config = ConfigDict(extra = "forbid", populate_by_name = True)


class StoryRandom(_Strict):
    weight: float = 1.0
    goto: str


class StoryEnding(_Strict):
    title: str
    kind: Literal["good", "bad", "neutral", "secret"] = "neutral"


class StoryScene(_Strict):
    id: str
    title: Optional[str] = None
    background: Optional[dict[str, Any]] = None
    background_prompt: Optional[str] = None
    music: Optional[dict[str, Any]] = None
    on_stage: list[str] = Field(default_factory = list)
    lines: list[StoryLine] = Field(default_factory = list)
    choices: list[StoryChoice] = Field(default_factory = list)
    next: Optional[str] = None
    random: list[StoryRandom] = Field(default_factory = list)
    on_enter: dict[str, Expr] = Field(default_factory = dict)
    ending: Optional[StoryEnding] = None
    generate_background: bool = False
    generate_lines: bool = False
    transition: Optional[StoryTransition] = None  # background change on entry; None = dissolve


class StoryManifest(_Strict):
    title: str
    start: str
    vars: dict[str, Any] = Field(default_factory = dict)
    characters: list[StoryCharacter] = Field(default_factory = list)
    scenes: list[StoryScene] = Field(default_factory = list)
    mode: Literal["fixed", "live", "infinite"] = "fixed"
    style: Optional[str] = None
    sounds: Optional[dict[str, dict[str, Any]]] = None
    entry_points: dict[str, str] = Field(default_factory = dict)


class WorldNpc(_Strict):
    id: str
    name: str
    persona: str = ""
    greeting: str = ""
    sprite: Optional[dict[str, Any]] = None
    sprite_prompt: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    wander: bool = True
    routine: Optional[str] = None
    relationships: Optional[dict[str, str]] = None
    seed_memories: Optional[list[dict[str, Any]]] = None  # [{text, importance}]
    role: Optional[Literal["rival", "hostile"]] = None


class WorldItem(_Strict):
    id: str
    name: str
    description: str = ""
    sprite: Optional[dict[str, Any]] = None
    sprite_prompt: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None


class WorldQuest(_Strict):
    id: str
    title: str
    description: str = ""
    giver: str
    requires_item: Optional[str] = None
    reward_text: str = ""
    set: dict[str, Expr] = Field(default_factory = dict)


class WorldRegion(_Strict):
    id: str
    name: str
    x: int
    y: int
    w: int
    h: int
    on_enter_text: str = ""


class WorldPlace(_Strict):
    id: str
    name: str
    x: int
    y: int
    w: int
    h: int


class WorldSimulation(_Strict):
    """Generative Agents style village life: a game clock, NPC memories, daily plans and NPC to NPC chats."""

    enabled: bool = True
    minutes_per_second: float = 2.0
    start_minute: int = 480
    places: list[WorldPlace] = Field(default_factory = list)
    chat_cooldown_min: int = 120  # game minutes per NPC pair
    calls_per_minute: int = 4  # background LLM calls per real minute, across every NPC
    needs: bool = True  # Sims-style needs pick what villagers do between plan blocks
    options: Optional[bool] = None  # option-action plans that run until a termination condition
    plan_cache: Optional[int] = (
        None  # LRU size of reusable plans keyed by role, needs, time and place
    )
    reflect_at: Optional[int] = (
        None  # reflect when summed importance since the last reflection crosses this
    )
    rate_batch: Optional[int] = None  # observations scored per importance call
    relationships: Optional[bool] = None  # pair affinity and facts, rumour source chains


class WorldAutonomy(_Strict):
    trade: bool = False  # make, buy and sell goods
    homes: bool = False  # claim a free house or build one
    quests: bool = False  # a rival races the player for quest items
    rivals: bool = False  # the quest director may add rivals
    enemies: bool = False  # hostile villagers chase and rob (non-lethal)


class WorldAction(_Strict):
    """Action play in the 3D view; every flag off keeps walking as it was."""

    movement: bool = False  # gravity, jumping, sprint, crouch and collisions with scenery
    shooter: bool = False  # a hitscan weapon, health and ammo; hostile villagers fight back
    verbs: Optional[bool] = None  # dash; with the shooter also melee and thrown rocks (3D only)
    ai: Optional[Literal["v2"]] = (
        None  # hostiles see, hear, path, flank and take cover (3D shooter)
    )
    barks: Optional[dict[str, list[str]]] = (
        None  # lines hostiles shout, by event: see, hear, lose, pain, flee, attack
    )


class World3dView(_Strict):
    time_of_day: Optional[Literal["dawn", "day", "dusk", "night"]] = None
    sky: Optional[str] = None  # "#rrggbb"; the time of day picks one when unset
    fog: Optional[float] = Field(default = None, ge = 0, le = 1)  # 0 = clear, 1 = thick
    camera_height: Optional[float] = Field(default = None, ge = 0.3, le = 4)  # eye height in tiles
    walk_speed: Optional[float] = Field(default = None, ge = 0.5, le = 20)  # tiles per second
    terrain: Optional[float] = Field(default = None, ge = 0, le = 3)  # hill height scale
    ambient: Optional[dict[str, Any]] = None  # an audio MediaValue looped while you play


class WorldProp3d(_Strict):
    """A piece of scenery placed in the 3D editor: a built-in low-poly prop or an imported GLB model."""

    id: str
    kind: Literal[
        "house",
        "tree",
        "lamp",
        "fence",
        "well",
        "crate",
        "rock",
        "bench",
        "stall",
        "door",
        "chest",
        "lever",
        "model",
    ]
    asset: Optional[dict[str, Any]] = None  # the GLB flow asset of a "model" prop
    name: Optional[str] = None
    x: float  # tile units; a prop snapped to a tile stands at its centre (x + 0.5)
    y: float
    rot: float = 0  # degrees about the vertical axis
    scale: float = 1
    solid: bool = True  # blocks walking on its tile, in 2D and 3D
    open: Optional[bool] = None  # a door, chest or lever that starts open (or pulled)
    locked_by: Optional[str] = None  # a door's key: the item id the player must carry
    contents: Optional[list[str]] = None  # item ids a chest gives, once
    targets: Optional[list[str]] = None  # door ids a lever opens and closes


class WorldFaction(_Strict):
    id: str
    name: str
    player: bool = False  # the faction the player runs; at most one
    leader: Optional[str] = None  # npc id who speaks for it
    region: Optional[str] = None  # home town region id
    persona: str = ""
    color: Optional[str] = None  # "#rrggbb"


class WorldRivals(_Strict):
    """Village Rivals: a turn-based 4X-lite where LLM-led villages compete on the world map."""

    factions: list[WorldFaction] = Field(min_length = 2, max_length = 4)
    max_turns: int = Field(default = 30, ge = 5, le = 200)
    hysteresis: float = Field(default = 0.15, ge = 0, le = 1)  # posture change needed to switch plans


class WorldCity(_Strict):
    """Mayor mode: the player zones and builds, a Micropolis-style simulation grows the town."""

    enabled: bool = True
    funds: int = Field(default = 20000, ge = 0, le = 1_000_000)
    tax: int = Field(default = 7, ge = 0, le = 20)
    advisor: bool = True  # a monthly advisor report from the chat model
    petitions: bool = True  # residents file 1-2 petitions a month
    events: bool = True  # the model may pick one event a month from a fixed menu


class WorldSpec(_Strict):
    title: str
    seed: int = 0
    width: int = Field(default = 48, ge = 8, le = 256)
    height: int = Field(default = 36, ge = 8, le = 256)
    tile_size: int = Field(default = 32, ge = 8, le = 128)
    biome: Literal["meadow", "forest", "desert", "snow", "town", "dungeon", "beach"] = "meadow"
    tileset: dict[str, dict[str, Any]] = Field(default_factory = dict)
    player: dict[str, Any] = Field(default_factory = lambda: {"name": "You"})
    npcs: list[WorldNpc] = Field(default_factory = list)
    items: list[WorldItem] = Field(default_factory = list)
    quests: list[WorldQuest] = Field(default_factory = list)
    regions: list[WorldRegion] = Field(default_factory = list)
    vars: dict[str, Any] = Field(default_factory = dict)
    intro_text: str = ""
    style: Optional[str] = None
    entry_points: dict[str, str] = Field(default_factory = dict)
    simulation: Optional[WorldSimulation] = None
    autonomy: Optional[WorldAutonomy] = None
    action: Optional[WorldAction] = None
    render: Optional[Literal["2d", "3d"]] = None
    view3d: Optional[World3dView] = None
    props3d: Optional[list[WorldProp3d]] = None
    rivals: Optional[WorldRivals] = None
    player_node: Optional[str] = None
    city: Optional[WorldCity] = None


RunStatus = Literal[
    "queued", "running", "cancelling", "cancelled", "completed", "failed", "interrupted"
]
NodeStatus = Literal[
    "idle",
    "queued",
    "waiting",
    "loading_model",
    "running",
    "done",
    "cached",
    "skipped",
    "failed",
    "cancelled",
]
EventType = Literal[
    "run.queued",
    "run.started",
    "run.completed",
    "run.failed",
    "run.cancelling",
    "run.cancelled",
    "run.interrupted",
    "node.status",  # data: {status, message?}
    "node.progress",  # data: {fraction?, step?, total?, phase?, preview?(data URL, small)}
    "node.output",  # data: {outputs: {port: value}, cached: bool}
    "node.log",  # data: {level, message}
    "node.text_delta",  # data: {port, delta}
    "snapshot",  # data: full run state, sent first on (re)connect
    "heartbeat",
]


class RunEvent(BaseModel):
    seq: int
    run_id: str
    type: EventType
    node_id: Optional[str] = None
    ts: float
    data: dict[str, Any] = Field(default_factory = dict)


JsonValue = Union[None, bool, int, float, str, list, dict]
