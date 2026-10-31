// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Mirror of studio/backend/core/flows/schema.py and specs.py; keep the two in sync.

export const FLOW_FORMAT = "unsloth-flow";
export const FLOW_FORMAT_VERSION = 1;

export type FlowKind = "pipeline" | "story" | "world";

export type PortType =
  | "text"
  | "number"
  | "bool"
  | "json"
  | "list"
  | "image"
  | "mask"
  | "audio"
  | "video"
  | "dataset"
  | "model"
  | "checkpoint"
  | "story"
  | "world"
  | "table"
  | "chart"
  | "heatmap"
  | "tensor"
  | "any"
  | "flow";

export type EdgeKind = "data" | "flow" | "feedback";

export interface FlowParam {
  id: string;
  type: "text" | "number" | "bool" | "json" | "image" | "audio";
  label?: string | null;
  default?: unknown;
  description?: string | null;
}

export interface ModelRequirement {
  ref: string;
  kind:
    | "chat"
    | "vision"
    | "image"
    | "video"
    | "audio"
    | "tts"
    | "stt"
    | "embedding";
  repo: string;
  variant?: string | null;
  optional?: boolean;
  note?: string | null;
}

export interface AssetRef {
  id: string; // sha256:<hex>
  mime: string;
  name?: string | null;
  bytes: number;
}

export interface FlowNode {
  id: string;
  type: string;
  v: number;
  params: Record<string, unknown>;
  pos: [number, number];
  label?: string | null;
  group?: string | null;
  collapsed?: boolean;
  disabled?: boolean;
}

export interface FlowEdge {
  id: string;
  source: string;
  source_port: string;
  target: string;
  target_port: string;
  kind: EdgeKind;
}

export interface FlowGroup {
  id: string;
  label: string;
  color?: string | null;
  rect: [number, number, number, number];
  /** "recipe": one Data Recipe; `config` is the recipe minus its columns (they are recipe.column nodes). */
  kind?: "recipe" | "loop" | null;
  config?: Record<string, unknown> | null;
  run?: Record<string, unknown> | null;
  ui?: Record<string, unknown> | null;
}

export interface FlowNote {
  id: string;
  text: string;
  rect: [number, number, number, number];
  color?: string | null;
}

export interface FlowDocument {
  format: typeof FLOW_FORMAT;
  version: number;
  id: string;
  name: string;
  description: string;
  kind: FlowKind;
  seed: number;
  params: FlowParam[];
  requirements: { models: ModelRequirement[] };
  nodes: FlowNode[];
  edges: FlowEdge[];
  groups: FlowGroup[];
  notes: FlowNote[];
  assets: AssetRef[];
  baked: Record<string, Record<string, unknown>>;
  editor: Record<string, unknown>;
  meta: Record<string, unknown>;
}

export type ParamKind =
  | "text"
  | "textarea"
  | "number"
  | "int"
  | "bool"
  | "select"
  | "model"
  | "seed"
  | "json"
  | "color"
  | "string_list"
  | "expr"
  | "asset"
  | "story"
  | "world"
  | "schema"
  | "lora_list";

export interface ParamSpec {
  id: string;
  kind: ParamKind;
  default: unknown;
  label: string;
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
  model_kind?: string;
  accept?: "image" | "audio" | "video" | "document";
  options_from?: string;
  options_path?: string;
  bindable?: boolean;
  advanced?: boolean;
}

export interface PortSpec {
  id: string;
  type: PortType;
  label: string;
  optional?: boolean;
  multi?: boolean;
  item?: PortType;
  param?: string;
}

export interface NodeSpec {
  type: string;
  v: number;
  title: string;
  category: string;
  description: string;
  inputs: PortSpec[];
  outputs: PortSpec[];
  params: ParamSpec[];
  lane: "cpu" | "gpu" | "player";
  model_kind: string | null;
  cacheable: boolean;
  side_effect: boolean;
  icon: string | null;
  player: boolean;
  entrypoint: boolean;
  flow_in: boolean;
  flow_out: string[];
  requires?: string[];
  exclusive_gpu?: boolean;
  requires_approval?: string[];
  /** Part of something another node runs (a recipe column); never run on its own. */
  design?: boolean;
}

export interface NodeCategory {
  id: string;
  title: string;
  color: string;
}

export interface NodeCatalog {
  version: number;
  categories: NodeCategory[];
  port_colors: Record<string, string>;
  nodes: NodeSpec[];
}

export interface MediaValue {
  kind: "image" | "audio" | "video";
  ref: string; // "image:<id>" | "audio:<id>" | "video:<id>" | "sha256:<hex>" | "flowasset:<id>"
  url?: string | null;
  mime?: string | null;
  width?: number | null;
  height?: number | null;
  duration_s?: number | null;
  seed?: number | null;
  prompt?: string | null;
}

export interface DatasetValue {
  kind: "dataset";
  source: "hf" | "recipe" | "local";
  ref: string;
  rows?: number | null;
  columns?: string[] | null;
  /** Which block wrote a local upload, for its label. */
  made_by?: "preview" | "recipe" | "rows" | null;
}

export interface ModelValue {
  kind: "model";
  model_kind: string;
  repo: string;
  variant?: string | null;
}

export interface CheckpointValue {
  kind: "checkpoint";
  name: string;
  base_model?: string | null;
}

// Model param value on a node: null = "use the loaded model".
export interface ModelChoice {
  repo: string;
  variant?: string | null;
}

export type Expr =
  | null
  | boolean
  | number
  | string
  | { var: string }
  | { op: ExprOp; args: Expr[] }
  | { list: Expr[] };

export const EXPR_OPS = [
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
] as const;
export type ExprOp = (typeof EXPR_OPS)[number];

export interface StoryCharacter {
  id: string;
  name: string;
  color?: string | null;
  look?: string | null;
  voice?: string | null;
  portrait?: MediaValue | null;
  expressions?: Record<string, MediaValue>;
  side?: "left" | "right" | "center";
}

export type StoryTransition =
  | "dissolve"
  | "fade"
  | "flash"
  | "slideleft"
  | "slideright"
  | "none";

export type StoryPosition =
  | "left"
  | "center"
  | "right"
  | "twoleft"
  | "tworight";

export interface StoryStage {
  show?: string | null;
  hide?: string | null;
  at?: StoryPosition | null;
  expression?: string | null;
  with?: StoryTransition | null;
  sound?: string | null;
  pause?: number | null;
}

export interface StoryLine {
  speaker?: string | null;
  text: string;
  expression?: string | null;
  if?: Expr;
  voice?: MediaValue | null;
  stage?: StoryStage | null;
}

export interface StoryChoice {
  id: string;
  text: string;
  goto: string;
  if?: Expr;
  set?: Record<string, Expr>;
}

export interface StoryScene {
  id: string;
  title?: string | null;
  background?: MediaValue | null;
  background_prompt?: string | null;
  music?: MediaValue | null;
  on_stage?: string[];
  lines: StoryLine[];
  choices: StoryChoice[];
  next?: string | null;
  random?: { weight: number; goto: string }[];
  on_enter?: Record<string, Expr>;
  ending?: {
    title: string;
    kind: "good" | "bad" | "neutral" | "secret";
  } | null;
  generate_background?: boolean;
  generate_lines?: boolean;
  transition?: StoryTransition | null;
}

export interface StoryManifest {
  title: string;
  start: string;
  vars: Record<string, unknown>;
  characters: StoryCharacter[];
  scenes: StoryScene[];
  mode: "fixed" | "live" | "infinite";
  style?: string | null;
  sounds?: Record<string, MediaValue> | null;
  entry_points?: Record<string, string>;
}

export type WorldBiome =
  | "meadow"
  | "forest"
  | "desert"
  | "snow"
  | "town"
  | "dungeon"
  | "beach";

export interface WorldNpc {
  id: string;
  name: string;
  persona: string;
  greeting: string;
  sprite?: MediaValue | null;
  sprite_prompt?: string | null;
  x?: number | null;
  y?: number | null;
  wander?: boolean;
  routine?: string;
  relationships?: Record<string, string>;
  seed_memories?: { text: string; importance: number }[];
  role?: "rival" | "hostile" | null;
}

export interface WorldItem {
  id: string;
  name: string;
  description: string;
  sprite?: MediaValue | null;
  sprite_prompt?: string | null;
  x?: number | null;
  y?: number | null;
}

export interface WorldQuest {
  id: string;
  title: string;
  description: string;
  giver: string;
  requires_item?: string | null;
  reward_text: string;
  set?: Record<string, Expr>;
}

export interface WorldRegion {
  id: string;
  name: string;
  x: number;
  y: number;
  w: number;
  h: number;
  on_enter_text: string;
}

export interface WorldPlace {
  id: string;
  name: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface WorldSimulation {
  enabled: boolean;
  minutes_per_second: number;
  start_minute: number;
  places: WorldPlace[];
  chat_cooldown_min: number;
  calls_per_minute: number;
  needs?: boolean;
  options?: boolean;
  plan_cache?: number;
  reflect_at?: number;
  rate_batch?: number;
  relationships?: boolean;
}

export interface WorldSpec {
  title: string;
  seed: number;
  width: number;
  height: number;
  tile_size: number;
  biome: WorldBiome;
  tileset: Record<string, MediaValue>;
  player: {
    name: string;
    sprite?: MediaValue | null;
    sprite_prompt?: string | null;
  };
  npcs: WorldNpc[];
  items: WorldItem[];
  quests: WorldQuest[];
  regions: WorldRegion[];
  vars: Record<string, unknown>;
  intro_text: string;
  style?: string | null;
  entry_points?: Record<string, string>;
  simulation?: WorldSimulation;
  summarise?: boolean;
  new_quests?: boolean;
  autonomy?: WorldAutonomy | null;
  action?: WorldAction | null;
  render?: "2d" | "3d";
  view3d?: World3dView;
  props3d?: WorldProp3d[];
  rivals?: WorldRivals | null;
  player_node?: string;
  city?: WorldCity | null;
}

export interface WorldCity {
  enabled?: boolean;
  funds?: number;
  tax?: number;
  advisor?: boolean;
  petitions?: boolean;
  events?: boolean;
}

export interface WorldFaction {
  id: string;
  name: string;
  player?: boolean;
  leader?: string | null;
  region?: string | null;
  persona?: string;
  color?: string | null;
}

export interface WorldRivals {
  factions: WorldFaction[];
  max_turns?: number;
  hysteresis?: number;
}

export type WorldPropKind =
  | "house"
  | "tree"
  | "lamp"
  | "fence"
  | "well"
  | "crate"
  | "rock"
  | "bench"
  | "stall"
  | "door"
  | "chest"
  | "lever"
  | "model";

export interface ModelAsset {
  kind: "model";
  ref: string;
  url?: string | null;
  mime?: string | null;
}

export interface WorldProp3d {
  id: string;
  kind: WorldPropKind;
  asset?: ModelAsset;
  name?: string;
  x: number;
  y: number;
  rot?: number;
  scale?: number;
  solid?: boolean;
  open?: boolean;
  locked_by?: string;
  contents?: string[];
  targets?: string[];
}

export interface World3dView {
  time_of_day?: "dawn" | "day" | "dusk" | "night";
  sky?: string;
  fog?: number;
  camera_height?: number;
  walk_speed?: number;
  terrain?: number;
  ambient?: MediaValue;
}

export interface WorldAction {
  movement?: boolean;
  shooter?: boolean;
  verbs?: boolean;
  ai?: "v2";
  barks?: Partial<Record<string, string[]>>;
}

export interface WorldAutonomy {
  trade?: boolean;
  homes?: boolean;
  quests?: boolean;
  rivals?: boolean;
  enemies?: boolean;
}

export type RunStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "cancelled"
  | "completed"
  | "failed"
  | "interrupted";
export type NodeStatus =
  | "idle"
  | "queued"
  | "waiting"
  | "loading_model"
  | "running"
  | "done"
  | "cached"
  | "skipped"
  | "failed"
  | "cancelled";

export type RunEventType =
  | "run.queued"
  | "run.started"
  | "run.completed"
  | "run.failed"
  | "run.cancelling"
  | "run.cancelled"
  | "run.interrupted"
  | "node.status"
  | "node.progress"
  | "node.output"
  | "node.log"
  | "node.text_delta"
  | "node.partial"
  | "snapshot"
  | "heartbeat";

export interface RunEvent {
  seq: number;
  run_id: string;
  type: RunEventType;
  node_id?: string | null;
  ts: number;
  data: Record<string, unknown>;
}

export interface FlowSummary {
  id: string;
  name: string;
  kind: FlowKind;
  description: string;
  updatedAt: number;
  createdAt: number;
  cover?: string | null;
  nodeCount: number;
  example?: boolean;
  tags?: string[];
  needs?: string[];
}

export interface FlowRecord {
  id: string;
  name: string;
  doc: FlowDocument;
  createdAt: number;
  updatedAt: number;
}

export interface FlowRunSummary {
  id: string;
  flow_id: string;
  status: RunStatus;
  started_at: number | null;
  finished_at: number | null;
  error?: string | null;
  outputs?: Record<string, unknown>;
}
