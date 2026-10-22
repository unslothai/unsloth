# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Flow document, the values that travel along edges, the story manifest and the world spec.

This module is the contract shared with the frontend (`studio/frontend/src/features/flows/types.ts`).
A Flow document is pure data: it names node types from the built-in registry and never carries code,
host paths, URLs to fetch, credentials or MCP server configs."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

FORMAT = "unsloth-flow"
FORMAT_VERSION = 1

# Limits enforced on save, import and run.
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

# Port data types. `any` connects to everything; `list` ports carry `item` in the port spec.
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

EdgeKind = Literal["data", "flow"]


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
    # Optional group id: a For Each node runs the nodes of its group once per item.
    group: Optional[str] = Field(default = None, max_length = 64)
    collapsed: bool = False
    # Disabled nodes are skipped (their outputs are absent).
    disabled: bool = False


class FlowEdge(_Strict):
    id: str = Field(pattern = r"^[A-Za-z0-9_-]{1,96}$")
    source: str
    source_port: str
    target: str
    target_port: str
    kind: EdgeKind = "data"


class FlowGroup(_Strict):
    id: str = Field(pattern = r"^[A-Za-z0-9_-]{1,64}$")
    label: str = Field(default = "Group", max_length = 200)
    color: Optional[str] = Field(default = None, pattern = r"^#[0-9a-fA-F]{6}$")
    rect: tuple[float, float, float, float] = (0.0, 0.0, 400.0, 300.0)


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
    # Pre-generated node outputs: {node_id: {port: value}}. Values may reference embedded assets.
    baked: dict[str, dict[str, Any]] = Field(default_factory = dict)
    # Editor-only state (viewport, layout direction); ignored by the engine.
    editor: dict[str, Any] = Field(default_factory = dict)
    # Free metadata (author, tags, cover asset id, created_with).
    meta: dict[str, Any] = Field(default_factory = dict)


# ---------------------------------------------------------------------------
# Values that travel along edges
# ---------------------------------------------------------------------------
# text -> str, number -> float|int, bool -> bool, json -> any JSON, list -> list
# image/audio/video -> MediaValue dict, dataset -> DatasetValue, model -> ModelValue,
# checkpoint -> CheckpointValue, story -> StoryManifest dict, world -> WorldSpec dict.


class MediaValue(_Strict):
    kind: Literal["image", "audio", "video"]
    # One of: a Studio gallery id ("image:<id>", "audio:<id>", "video:<id>"), an embedded asset ("sha256:<hex>"),
    # or a flow output stored by the engine ("flowasset:<id>").
    ref: str
    url: Optional[str] = None  # same-origin, authenticated URL the UI can load
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
    name: str  # "<run folder>" or "<run folder>/checkpoint-N", as MCP names them
    base_model: Optional[str] = None


# ---------------------------------------------------------------------------
# Expressions: a JSON tree, evaluated by core/flows/expr.py and src/features/flows/expr.ts
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Story manifest (consumed by the story player)
# ---------------------------------------------------------------------------
class StoryCharacter(_Strict):
    id: str
    name: str
    color: Optional[str] = None
    look: Optional[str] = None  # appearance prompt
    voice: Optional[str] = None
    portrait: Optional[dict[str, Any]] = None  # MediaValue
    expressions: dict[str, dict[str, Any]] = Field(default_factory = dict)  # name -> MediaValue
    side: Literal["left", "right", "center"] = "left"


class StoryLine(_Strict):
    speaker: Optional[str] = None  # character id, None = narrator
    text: str
    expression: Optional[str] = None
    cond: Expr = Field(default = None, alias = "if")
    voice: Optional[dict[str, Any]] = None  # MediaValue (pre-rendered narration)

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
    background: Optional[dict[str, Any]] = None  # MediaValue
    background_prompt: Optional[str] = None
    music: Optional[dict[str, Any]] = None  # MediaValue
    on_stage: list[str] = Field(default_factory = list)  # character ids
    lines: list[StoryLine] = Field(default_factory = list)
    choices: list[StoryChoice] = Field(default_factory = list)
    next: Optional[str] = None
    random: list[StoryRandom] = Field(default_factory = list)
    on_enter: dict[str, Expr] = Field(default_factory = dict)  # var -> Expr
    ending: Optional[StoryEnding] = None
    # Live generation: the player asks the engine to fill these when the scene is entered.
    generate_background: bool = False
    generate_lines: bool = False


class StoryManifest(_Strict):
    title: str
    start: str
    vars: dict[str, Any] = Field(default_factory = dict)
    characters: list[StoryCharacter] = Field(default_factory = list)
    scenes: list[StoryScene] = Field(default_factory = list)
    # "infinite": the player asks the flow's `next_scene` entrypoint for a new scene after every choice.
    mode: Literal["fixed", "live", "infinite"] = "fixed"
    style: Optional[str] = None  # art style appended to background/portrait prompts
    entry_points: dict[str, str] = Field(
        default_factory = dict
    )  # name -> node id of a subgraph output


# ---------------------------------------------------------------------------
# World spec (consumed by the world player)
# ---------------------------------------------------------------------------
class WorldNpc(_Strict):
    id: str
    name: str
    persona: str = ""
    greeting: str = ""
    sprite: Optional[dict[str, Any]] = None  # MediaValue
    sprite_prompt: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    wander: bool = True


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
    giver: str  # npc id
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


class WorldSpec(_Strict):
    title: str
    seed: int = 0
    width: int = Field(default = 48, ge = 8, le = 256)
    height: int = Field(default = 36, ge = 8, le = 256)
    tile_size: int = Field(default = 32, ge = 8, le = 128)
    biome: Literal["meadow", "forest", "desert", "snow", "town", "dungeon", "beach"] = "meadow"
    # Tile kind -> MediaValue. Missing kinds are drawn procedurally.
    tileset: dict[str, dict[str, Any]] = Field(default_factory = dict)
    player: dict[str, Any] = Field(default_factory = lambda: {"name": "You"})
    npcs: list[WorldNpc] = Field(default_factory = list)
    items: list[WorldItem] = Field(default_factory = list)
    quests: list[WorldQuest] = Field(default_factory = list)
    regions: list[WorldRegion] = Field(default_factory = list)
    vars: dict[str, Any] = Field(default_factory = dict)
    intro_text: str = ""
    style: Optional[str] = None
    entry_points: dict[str, str] = Field(default_factory = dict)  # e.g. {"npc_reply": "<node id>"}


# ---------------------------------------------------------------------------
# Run events (SSE `data:` payloads; `id:` is `seq`)
# ---------------------------------------------------------------------------
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
