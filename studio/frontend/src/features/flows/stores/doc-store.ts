// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { renameRoot, replaceRef } from "@/features/recipe-studio/utils/refs";
import { foldedGroups, readSuperCards } from "../editor/super-card-records";
import { referenceEdges } from "../recipe/recipe-columns";
import {
  FLOW_FORMAT,
  FLOW_FORMAT_VERSION,
  type FlowDocument,
  type FlowEdge,
  type FlowGroup,
  type FlowNode,
  type FlowNote,
} from "../types";
import {
  type FlowReroute,
  REROUTE_IN,
  REROUTE_OUT,
  type RerouteWire,
  rerouteUpstream,
  reroutesFromEditor,
  reroutesToEditor,
  syncRerouteEdges,
} from "./reroute-graph";

/** recipe-columns.ts GENERATE_TYPE; kept here so the store imports nothing from the recipe editor. */
const RECIPE_DATASET = "recipe.generate";
/** recipe-columns.ts COLUMN_TYPE. */
const RECIPE_COLUMN = "recipe.column";

/** A column or Recipe Dataset step that runs only in its recipe: framing or grouping never moves it out. */
function recipeOwned(node: FlowNode, groups: Record<string, FlowGroup>): boolean {
  return (
    (node.type === RECIPE_COLUMN || node.type === RECIPE_DATASET) &&
    !!node.group &&
    groups[node.group]?.kind === "recipe"
  );
}

/** A recipe column's name, from its params. */
function columnNameOf(node: FlowNode): string | null {
  const column = (node.params as { column?: { name?: unknown } } | undefined)?.column;
  return typeof column?.name === "string" ? column.name : null;
}

/** A column pasted or duplicated into a recipe that already has its name gets the next free one,
 * since a recipe's column names are unique. */
function uniqueColumn(
  node: FlowNode,
  group: string | null,
  nodes: Record<string, FlowNode>,
  order: string[],
  config?: Record<string, unknown>,
): FlowNode {
  const name = columnNameOf(node);
  if (node.type !== RECIPE_COLUMN || !group || !name) return node;
  const taken = new Set<unknown>([
    ...order
      .filter((id) => nodes[id].type === RECIPE_COLUMN && nodes[id].group === group)
      .map((id) => columnNameOf(nodes[id])),
    // Columns share one namespace with the recipe's other blocks: two of one name never open again.
    ...setupNames(config),
  ]);
  if (!taken.has(name)) return node;
  let n = 2;
  while (taken.has(`${name}_${n}`)) n += 1;
  const next = `${name}_${n}`;
  const params = node.params as { column: Record<string, unknown> };
  return {
    ...node,
    label: next,
    params: { ...node.params, column: { ...params.column, name: next } },
  };
}

/** The names a recipe's providers, model presets, tool profiles and seed take. */
function setupNames(config: Record<string, unknown> | undefined): unknown[] {
  if (!config) return [];
  const pick = (key: string, field: string) =>
    Array.isArray(config[key])
      ? (config[key] as Record<string, unknown>[]).map((item) => item?.[field])
      : [];
  return [
    ...pick("model_providers", "name"),
    ...pick("mcp_providers", "name"),
    ...pick("model_configs", "alias"),
    ...pick("tool_configs", "tool_alias"),
    ...(config.seed_config ? ["seed"] : []),
  ];
}

/** Fields that name another column outright, not through a Jinja reference (`column_name` is an
 * AI column's image context). */
const COLUMN_NAME_FIELDS = new Set([
  "category",
  "reference_column_name",
  "subcategory_parent",
  "column_name",
]);

/** A column's definition with column ``from`` renamed to ``to`` wherever it reads it: Jinja
 * references, the fields that name a column, a validator's target list and sampler conditions. */
function renameInColumn(value: unknown, from: string, to: string, key = ""): unknown {
  if (typeof value === "string")
    return COLUMN_NAME_FIELDS.has(key) && value === from ? to : replaceRef(value, from, to);
  if (Array.isArray(value))
    return key === "target_columns"
      ? value.map((v) => (v === from ? to : v))
      : value.map((v) => renameInColumn(v, from, to));
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value).map(([k, v]) => [
        key === "conditional_params" ? renameRoot(k, from, to) : k,
        renameInColumn(v, from, to, k),
      ]),
    );
  return value;
}

/** Graph state normalized by id. Every record is immutable: an edit replaces only what it touches,
 *  so a node component selecting `nodes[id]` re-renders only when that node changed. */
export interface DocGraph {
  nodes: Record<string, FlowNode>;
  nodeOrder: string[];
  edges: Record<string, FlowEdge>;
  edgeOrder: string[];
  groups: Record<string, FlowGroup>;
  groupOrder: string[];
  notes: Record<string, FlowNote>;
  noteOrder: string[];
  reroutes: Record<string, FlowReroute>;
  rerouteOrder: string[];
  wires: Record<string, RerouteWire>;
  wireOrder: string[];
}

export type DocMeta = Omit<
  FlowDocument,
  "nodes" | "edges" | "groups" | "notes"
>;

interface Snapshot {
  graph: DocGraph;
  meta: DocMeta;
}

export interface GraphFragment {
  nodes: FlowNode[];
  edges: FlowEdge[];
  /** Each copied node's frame where it was copied from: a pasted reader is renamed only after
   * copies from its own recipe. */
  from?: Record<string, string | null>;
}

export interface GraphPatch {
  nodes?: FlowNode[];
  groups?: FlowGroup[];
  addEdges?: FlowEdge[];
  removeEdges?: string[];
}

export interface DocState {
  flowId: string | null;
  meta: DocMeta;
  graph: DocGraph;
  version: number;
  past: Snapshot[];
  future: Snapshot[];
  lastCoalesceKey: string | null;
  lastCommitAt: number;

  load: (doc: FlowDocument) => void;
  reset: () => void;
  setMeta: (patch: Partial<DocMeta>, coalesceKey?: string) => void;
  addNode: (node: FlowNode) => void;
  updateParams: (
    id: string,
    patch: Record<string, unknown>,
    coalesceKey?: string,
  ) => void;
  updateNode: (
    id: string,
    patch: Partial<Omit<FlowNode, "id" | "params">>,
  ) => void;
  moveNodes: (
    positions: Record<string, [number, number]>,
    membership?: Record<string, string | null>,
  ) => void;
  removeElements: (
    ids: {
      nodes?: string[];
      edges?: string[];
      groups?: string[];
      notes?: string[];
      reroutes?: string[];
    },
    coalesceKey?: string,
  ) => void;
  connect: (edge: FlowEdge, replaceSingleInput: boolean) => void;
  /** Adds (or moves) a reroute and threads it into `splitId`, a plain edge or a wire segment. */
  addReroute: (
    reroute: FlowReroute,
    splitId?: string,
    style?: RerouteWire["style"],
  ) => void;
  /** Replaces existing reroutes (turning one keeps its centre, so its box moves too). */
  replaceReroutes: (reroutes: FlowReroute[]) => void;
  /** Connects a wire with a reroute on at least one end. */
  connectWire: (wire: RerouteWire, replaceSingleInput: boolean) => void;
  insertFragment: (fragment: GraphFragment) => void;
  upsertGroup: (group: FlowGroup) => void;
  resizeGroup: (group: FlowGroup) => void;
  groupNodes: (group: FlowGroup, nodeIds: string[]) => void;
  upsertNote: (note: FlowNote, coalesceKey?: string) => void;
  /** One undo step that upserts nodes and groups and adds or drops edges (a recipe group edit). */
  patchGraph: (patch: GraphPatch, coalesceKey?: string) => void;
  /** Tidy up's result as one undo step: card spots and sizes, frames, notes and reroutes. */
  applyTidy: (patch: TidyPatch) => void;
  undo: () => void;
  redo: () => void;
}

export interface TidyPatch {
  nodes: Record<string, { pos: [number, number]; size?: [number, number] | null }>;
  groups: Record<string, [number, number, number, number]>;
  notes: Record<string, [number, number, number, number]>;
  reroutes: Record<string, [number, number]>;
}

export const HISTORY_LIMIT = 100;
export const GROUP_KEY = "group:";
export const NOTE_KEY = "note:";
const COALESCE_MS = 900;

export function emptyGraph(): DocGraph {
  return {
    nodes: {},
    nodeOrder: [],
    edges: {},
    edgeOrder: [],
    groups: {},
    groupOrder: [],
    notes: {},
    noteOrder: [],
    reroutes: {},
    rerouteOrder: [],
    wires: {},
    wireOrder: [],
  };
}

export function emptyMeta(id = "untitled"): DocMeta {
  return {
    format: FLOW_FORMAT,
    version: FLOW_FORMAT_VERSION,
    id,
    name: "Untitled flow",
    description: "",
    kind: "pipeline",
    seed: 0,
    params: [],
    requirements: { models: [] },
    assets: [],
    baked: {},
    editor: {},
    meta: {},
  };
}

function byId<T extends { id: string }>(
  items: T[],
): [Record<string, T>, string[]] {
  const record: Record<string, T> = {};
  const order: string[] = [];
  for (const item of items) {
    if (record[item.id]) continue;
    record[item.id] = item;
    order.push(item.id);
  }
  return [record, order];
}

export function graphFromDocument(doc: FlowDocument): DocGraph {
  const [nodes, nodeOrder] = byId(doc.nodes ?? []);
  const [edges, edgeOrder] = byId(
    (doc.edges ?? []).filter((e) => nodes[e.source] && nodes[e.target]),
  );
  const [groups, groupOrder] = byId(doc.groups ?? []);
  const [notes, noteOrder] = byId(doc.notes ?? []);
  const graph: DocGraph = {
    nodes,
    nodeOrder,
    edges,
    edgeOrder,
    groups,
    groupOrder,
    notes,
    noteOrder,
    ...reroutesFromEditor(doc.editor, nodes),
  };
  if (!graph.wireOrder.length) return graph;
  // A wire whose plain edge was deleted outside the editor goes with it.
  const wires = { ...graph.wires };
  for (const id of graph.wireOrder) {
    const wire = wires[id];
    if (graph.reroutes[wire.target] || edges[id]) continue;
    if (rerouteUpstream(graph, wire.source)) delete wires[id];
  }
  return syncRerouteEdges({
    ...graph,
    wires,
    wireOrder: graph.wireOrder.filter((id) => wires[id]),
  });
}

export function metaFromDocument(doc: FlowDocument): DocMeta {
  const { nodes: _n, edges: _e, groups: _g, notes: _no, ...meta } = doc;
  const {
    reroutes: _r,
    reroute_wires: _w,
    ...editor
  } = (meta.editor ?? {}) as Record<string, unknown>;
  return {
    ...emptyMeta(doc.id),
    ...meta,
    editor,
    format: FLOW_FORMAT,
    requirements: meta.requirements ?? { models: [] },
  };
}

export function documentFromState(
  state: Pick<DocState, "meta" | "graph">,
): FlowDocument {
  const { meta, graph } = state;
  const reroutes = reroutesToEditor(graph);
  return {
    ...meta,
    ...(reroutes.reroutes || reroutes.reroute_wires
      ? { editor: { ...meta.editor, ...reroutes } }
      : {}),
    nodes: graph.nodeOrder.map((id) => graph.nodes[id]),
    edges: graph.edgeOrder.map((id) => graph.edges[id]),
    groups: graph.groupOrder.map((id) => graph.groups[id]),
    notes: graph.noteOrder.map((id) => graph.notes[id]),
  };
}

export function documentSignature(doc: FlowDocument): string {
  const canonical = documentFromState({
    meta: metaFromDocument(doc),
    graph: graphFromDocument(doc),
  });
  const { editor, ...rest } = canonical;
  // The viewport is not an edit; reroutes and a collapsed super card are.
  return JSON.stringify({
    ...rest,
    reroutes: editor.reroutes,
    reroute_wires: editor.reroute_wires,
    super_cards: editor.super_cards,
  });
}

let idCounter = 0;
export function newId(prefix: string): string {
  idCounter = (idCounter + 1) % 1_000_000;
  const random = Math.random().toString(36).slice(2, 8);
  return `${prefix}_${Date.now().toString(36).slice(-4)}${random}${idCounter.toString(36)}`;
}

export const useDocStore = create<DocState>((set, get) => {
  function commit(
    updater: (
      state: DocState,
    ) => Partial<Pick<DocState, "graph" | "meta">> | null,
    coalesceKey?: string,
  ): void {
    const state = get();
    const next = updater(state);
    if (!next) return;
    const now = Date.now();
    const coalesce =
      coalesceKey !== undefined &&
      coalesceKey === state.lastCoalesceKey &&
      now - state.lastCommitAt < COALESCE_MS;
    const past = coalesce
      ? state.past
      : [...state.past, { graph: state.graph, meta: state.meta }].slice(
          -HISTORY_LIMIT,
        );
    set({
      ...next,
      past,
      future: [],
      version: state.version + 1,
      lastCoalesceKey: coalesceKey ?? null,
      lastCommitAt: now,
    });
  }

  function withGraph(
    state: DocState,
    patch: Partial<DocGraph>,
  ): { graph: DocGraph } {
    return { graph: { ...state.graph, ...patch } };
  }

  return {
    flowId: null,
    meta: emptyMeta(),
    graph: emptyGraph(),
    version: 0,
    past: [],
    future: [],
    lastCoalesceKey: null,
    lastCommitAt: 0,

    load: (doc) =>
      set({
        flowId: doc.id,
        meta: metaFromDocument(doc),
        graph: graphFromDocument(doc),
        version: 0,
        past: [],
        future: [],
        lastCoalesceKey: null,
        lastCommitAt: 0,
      }),

    reset: () =>
      set({
        flowId: null,
        meta: emptyMeta(),
        graph: emptyGraph(),
        version: 0,
        past: [],
        future: [],
        lastCoalesceKey: null,
      }),

    setMeta: (patch, coalesceKey) =>
      commit((s) => ({ meta: { ...s.meta, ...patch } }), coalesceKey),

    addNode: (node) =>
      commit((s) =>
        withGraph(s, {
          nodes: {
            ...s.graph.nodes,
            [node.id]: {
              ...node,
              group: node.group ?? frameAt(s.graph, node.pos, foldedGroups(s.meta.editor, s.graph)),
            },
          },
          nodeOrder: [...s.graph.nodeOrder, node.id],
        }),
      ),

    updateParams: (id, patch, coalesceKey) =>
      commit((s) => {
        const node = s.graph.nodes[id];
        if (!node) return null;
        return withGraph(s, {
          nodes: {
            ...s.graph.nodes,
            [id]: { ...node, params: { ...node.params, ...patch } },
          },
        });
      }, coalesceKey),

    updateNode: (id, patch) =>
      commit(
        (s) => {
          const node = s.graph.nodes[id];
          if (!node) return null;
          return withGraph(s, {
            nodes: { ...s.graph.nodes, [id]: { ...node, ...patch } },
          });
        },
        patch.label !== undefined ? `label:${id}` : undefined,
      ),

    moveNodes: (positions, membership) =>
      commit((s) => {
        let nodes: Record<string, FlowNode> | null = null;
        let groups: Record<string, FlowGroup> | null = null;
        let notes: Record<string, FlowNote> | null = null;
        let reroutes: Record<string, FlowReroute> | null = null;
        for (const [key, pos] of Object.entries(positions)) {
          const reroute = s.graph.reroutes[key];
          if (reroute) {
            if (reroute.pos[0] === pos[0] && reroute.pos[1] === pos[1])
              continue;
            reroutes ??= { ...s.graph.reroutes };
            reroutes[key] = { ...reroute, pos };
            continue;
          }
          if (key.startsWith(GROUP_KEY)) {
            const group = s.graph.groups[key.slice(GROUP_KEY.length)];
            if (
              !group ||
              (group.rect[0] === pos[0] && group.rect[1] === pos[1])
            )
              continue;
            groups ??= { ...s.graph.groups };
            groups[group.id] = {
              ...group,
              rect: [pos[0], pos[1], group.rect[2], group.rect[3]],
            };
            continue;
          }
          if (key.startsWith(NOTE_KEY)) {
            const note = s.graph.notes[key.slice(NOTE_KEY.length)];
            if (!note || (note.rect[0] === pos[0] && note.rect[1] === pos[1]))
              continue;
            notes ??= { ...s.graph.notes };
            notes[note.id] = {
              ...note,
              rect: [pos[0], pos[1], note.rect[2], note.rect[3]],
            };
            continue;
          }
          const node = s.graph.nodes[key];
          if (!node || (node.pos[0] === pos[0] && node.pos[1] === pos[1]))
            continue;
          nodes ??= { ...s.graph.nodes };
          nodes[key] = { ...node, pos };
        }
        for (const [id, group] of Object.entries(membership ?? {})) {
          const node = (nodes ?? s.graph.nodes)[id];
          if (!node || (node.group ?? null) === group) continue;
          nodes ??= { ...s.graph.nodes };
          nodes[id] = { ...node, group };
        }
        if (!nodes && !groups && !notes && !reroutes) return null;
        return withGraph(s, {
          ...(nodes ? { nodes } : {}),
          ...(groups ? { groups } : {}),
          ...(notes ? { notes } : {}),
          ...(reroutes ? { reroutes } : {}),
        });
      }),

    removeElements: (ids, coalesceKey) =>
      commit((s) => {
        const dropNodes = new Set(ids.nodes ?? []);
        const dropEdges = new Set(ids.edges ?? []);
        const dropGroups = new Set(ids.groups ?? []);
        const dropNotes = new Set(ids.notes ?? []);
        const dropReroutes = new Set(
          (ids.reroutes ?? []).filter((id) => s.graph.reroutes[id]),
        );
        // React Flow sends a deleted node's wires along with it; a reroute's own wires heal instead.
        for (const id of dropEdges) {
          const wire = s.graph.wires[id];
          if (
            wire &&
            (dropReroutes.has(wire.source) || dropReroutes.has(wire.target))
          )
            dropEdges.delete(id);
        }
        for (const edgeId of s.graph.edgeOrder) {
          const edge = s.graph.edges[edgeId];
          if (dropNodes.has(edge.source) || dropNodes.has(edge.target))
            dropEdges.add(edgeId);
        }
        if (
          !dropNodes.size &&
          !dropEdges.size &&
          !dropGroups.size &&
          !dropNotes.size &&
          !dropReroutes.size
        )
          return null;
        const rewired = removeReroutes(
          s.graph,
          dropReroutes,
          (id) => dropEdges.has(id),
          (id) => dropNodes.has(id),
        );
        const omit = <T>(record: Record<string, T>, drop: Set<string>) => {
          if (!drop.size) return record;
          const next = { ...record };
          for (const id of drop) delete next[id];
          return next;
        };
        let nodes = omit(s.graph.nodes, dropNodes);
        if (dropGroups.size) {
          for (const [id, n] of Object.entries(nodes)) {
            if (n.group && dropGroups.has(n.group)) {
              if (nodes === s.graph.nodes) nodes = { ...nodes };
              nodes[id] = { ...n, group: null };
            }
          }
        }
        const graph = withGraph(s, {
          nodes,
          nodeOrder: dropNodes.size
            ? s.graph.nodeOrder.filter((id) => !dropNodes.has(id))
            : s.graph.nodeOrder,
          edges: omit(s.graph.edges, dropEdges),
          edgeOrder: dropEdges.size
            ? s.graph.edgeOrder.filter((id) => !dropEdges.has(id))
            : s.graph.edgeOrder,
          groups: omit(s.graph.groups, dropGroups),
          groupOrder: dropGroups.size
            ? s.graph.groupOrder.filter((id) => !dropGroups.has(id))
            : s.graph.groupOrder,
          notes: omit(s.graph.notes, dropNotes),
          noteOrder: dropNotes.size
            ? s.graph.noteOrder.filter((id) => !dropNotes.has(id))
            : s.graph.noteOrder,
          ...rewired.patch,
        }).graph;
        return {
          graph: syncRerouteEdges(
            graph,
            s.graph.wireOrder.filter((id) => !rewired.healed.has(id)),
          ),
        };
      }, coalesceKey),

    connect: (edge, replaceSingleInput) =>
      commit((s) => {
        const edges = { ...s.graph.edges };
        let order = s.graph.edgeOrder;
        for (const id of s.graph.edgeOrder) {
          const existing = s.graph.edges[id];
          const duplicate =
            existing.source === edge.source &&
            existing.source_port === edge.source_port &&
            existing.target === edge.target &&
            existing.target_port === edge.target_port;
          if (duplicate) return null;
          const occupiesInput =
            replaceSingleInput &&
            edge.kind === "data" &&
            existing.kind === "data" &&
            existing.target === edge.target &&
            existing.target_port === edge.target_port;
          const occupiesFlowOut =
            edge.kind === "flow" &&
            existing.kind === "flow" &&
            existing.source === edge.source &&
            existing.source_port === edge.source_port;
          if (occupiesInput || occupiesFlowOut) {
            delete edges[id];
            order = order.filter((other) => other !== id);
          }
        }
        edges[edge.id] = edge;
        const graph = withGraph(s, {
          edges,
          edgeOrder: [...order, edge.id],
        }).graph;
        // A segment out of an empty reroute has no edge yet, but still holds the input it ends at.
        const dropped = s.graph.wireOrder.filter((id) => {
          const wire = s.graph.wires[id];
          return (
            (s.graph.edges[id] && !edges[id]) ||
            (replaceSingleInput &&
              edge.kind === "data" &&
              wire.target === edge.target &&
              wire.target_port === edge.target_port)
          );
        });
        if (!dropped.length) return { graph };
        const wires = { ...graph.wires };
        for (const id of dropped) delete wires[id];
        return {
          graph: syncRerouteEdges(
            {
              ...graph,
              wires,
              wireOrder: graph.wireOrder.filter((id) => wires[id]),
            },
            s.graph.wireOrder,
          ),
        };
      }),

    addReroute: (reroute, splitId, style) =>
      commit((s) => {
        if (s.graph.nodes[reroute.id]) return null;
        // A recipe column's wire is a reference in its reader's prompt: it never goes through a reroute.
        const split = splitId ? (s.graph.wires[splitId] ?? s.graph.edges[splitId]) : undefined;
        if (
          split &&
          (s.graph.nodes[split.source]?.type === RECIPE_COLUMN ||
            s.graph.nodes[split.target]?.type === RECIPE_COLUMN)
        )
          return null;
        const exists = !!s.graph.reroutes[reroute.id];
        let graph: DocGraph = {
          ...s.graph,
          reroutes: { ...s.graph.reroutes, [reroute.id]: reroute },
          rerouteOrder: exists
            ? s.graph.rerouteOrder
            : [...s.graph.rerouteOrder, reroute.id],
        };
        const wire = splitId ? s.graph.wires[splitId] : undefined;
        const plain = splitId && !wire ? s.graph.edges[splitId] : undefined;
        const cut = wire ?? (plain?.kind === "data" ? plain : undefined);
        if (cut) {
          const pinned = style ?? wire?.style;
          const pin = pinned ? { style: pinned } : {};
          // The far half keeps the id, so the plain edge the compiler sees keeps its id too.
          const near: RerouteWire = {
            id: newId("w"),
            source: cut.source,
            source_port: cut.source_port,
            target: reroute.id,
            target_port: REROUTE_IN,
            ...pin,
          };
          const far: RerouteWire = {
            id: cut.id,
            source: reroute.id,
            source_port: REROUTE_OUT,
            target: cut.target,
            target_port: cut.target_port,
            ...pin,
          };
          const wires = { ...graph.wires, [near.id]: near, [far.id]: far };
          const wireOrder = wire
            ? graph.wireOrder.flatMap((id) =>
                id === wire.id ? [near.id, id] : [id],
              )
            : [...graph.wireOrder, near.id, far.id];
          graph = { ...graph, wires, wireOrder };
          graph = syncRerouteEdges(graph, s.graph.wireOrder);
        }
        return { graph };
      }),

    applyTidy: (patch) =>
      commit((s) => {
        const g = s.graph;
        const nodes = { ...g.nodes };
        for (const [id, next] of Object.entries(patch.nodes)) {
          const node = nodes[id];
          if (!node) continue;
          nodes[id] = {
            ...node,
            pos: next.pos,
            ...(next.size === undefined ? {} : { size: next.size }),
          };
        }
        const groups = { ...g.groups };
        for (const [id, rect] of Object.entries(patch.groups))
          if (groups[id]) groups[id] = { ...groups[id], rect };
        const notes = { ...g.notes };
        for (const [id, rect] of Object.entries(patch.notes))
          if (notes[id]) notes[id] = { ...notes[id], rect };
        const reroutes = { ...g.reroutes };
        for (const [id, pos] of Object.entries(patch.reroutes))
          if (reroutes[id]) reroutes[id] = { ...reroutes[id], pos };
        return { graph: { ...g, nodes, groups, notes, reroutes } };
      }),

    replaceReroutes: (list) =>
      commit((s) => {
        const known = list.filter((r) => s.graph.reroutes[r.id]);
        if (!known.length) return null;
        const reroutes = { ...s.graph.reroutes };
        for (const reroute of known) reroutes[reroute.id] = reroute;
        return { graph: { ...s.graph, reroutes } };
      }),

    connectWire: (wire, replaceSingleInput) =>
      commit((s) => {
        const g = s.graph;
        for (const id of g.wireOrder) {
          const w = g.wires[id];
          if (
            w.source === wire.source &&
            w.source_port === wire.source_port &&
            w.target === wire.target &&
            w.target_port === wire.target_port
          )
            return null;
        }
        const intoReroute = !!g.reroutes[wire.target];
        const occupies = (target: string, port: string) =>
          target === wire.target &&
          port === wire.target_port &&
          (intoReroute || replaceSingleInput);
        const wires: Record<string, RerouteWire> = {};
        const wireOrder: string[] = [];
        for (const id of g.wireOrder) {
          const w = g.wires[id];
          if (occupies(w.target, w.target_port)) continue;
          wires[id] = w;
          wireOrder.push(id);
        }
        wires[wire.id] = wire;
        wireOrder.push(wire.id);
        const edges = { ...g.edges };
        let edgeOrder = g.edgeOrder;
        if (!intoReroute) {
          for (const id of g.edgeOrder) {
            const e = g.edges[id];
            if (
              e.kind === "data" &&
              !g.wires[id] &&
              occupies(e.target, e.target_port)
            ) {
              delete edges[id];
              edgeOrder = edgeOrder.filter((other) => other !== id);
            }
          }
        }
        const graph = syncRerouteEdges(
          { ...g, wires, wireOrder, edges, edgeOrder },
          g.wireOrder,
        );
        // A multi input keeps every edge, so a rerouted wire must not repeat one already there.
        const seen = new Map<string, string>();
        for (const id of graph.edgeOrder) {
          const e = graph.edges[id];
          if (e.kind !== "data") continue;
          const key = `${e.source}\u0000${e.source_port}\u0000${e.target}\u0000${e.target_port}`;
          const other = seen.get(key);
          if (other && (graph.wires[id] || graph.wires[other])) return null;
          seen.set(key, id);
        }
        return { graph };
      }),

    insertFragment: (fragment) =>
      commit((s) => {
        const nodes = { ...s.graph.nodes };
        const nodeOrder = [...s.graph.nodeOrder];
        const folded = foldedGroups(s.meta.editor, s.graph);
        // A copied column renamed to stay unique is renamed in the other copies that read it too.
        const renames = new Map<string, [string, string][]>();
        const scopes = new Map<string, string>();
        const added: string[] = [];
        for (const node of fragment.nodes) {
          if (nodes[node.id]) continue;
          const group = frameAt(s.graph, node.pos, folded);
          // A recipe's blocks mean nothing outside a recipe frame (they carry none of its models or
          // seed), and one alone would stop the whole flow compiling: such a paste skips them.
          if (
            (node.type === RECIPE_COLUMN || node.type === RECIPE_DATASET) &&
            s.graph.groups[group ?? ""]?.kind !== "recipe"
          )
            continue;
          // A recipe runs from one Recipe Dataset step: a pasted or duplicated second one comes in off.
          const extra =
            node.type === RECIPE_DATASET &&
            !node.disabled &&
            !!group &&
            nodeOrder.some(
              (id) =>
                nodes[id].type === RECIPE_DATASET &&
                nodes[id].group === group &&
                !nodes[id].disabled,
            );
          const unique = uniqueColumn(
            node,
            group,
            nodes,
            nodeOrder,
            group ? (s.graph.groups[group]?.config ?? undefined) : undefined,
          );
          const from = columnNameOf(node);
          const to = columnNameOf(unique);
          const scope = `${fragment.from?.[node.id] ?? ""}>${group}`;
          if (group && from && to && from !== to)
            renames.set(scope, [...(renames.get(scope) ?? []), [from, to]]);
          scopes.set(node.id, scope);
          nodes[node.id] = { ...unique, group, ...(extra ? { disabled: true } : {}) };
          nodeOrder.push(node.id);
          added.push(node.id);
        }
        for (const id of added) {
          const node = nodes[id];
          const scope = scopes.get(id);
          const pairs = node.group && scope ? renames.get(scope) : undefined;
          if (node.type !== RECIPE_COLUMN || !pairs) continue;
          const own = columnNameOf(node);
          let column = (node.params as { column: Record<string, unknown> }).column;
          // Through placeholders, so a new name that is another copy's old name (topic -> topic_2 while
          // topic_2 -> topic_2_2) is not renamed a second time.
          const held = pairs.map((_, i) => `__flows_paste_${i}__`);
          pairs.forEach(([from], i) => {
            column = renameInColumn(column, from, held[i]) as Record<string, unknown>;
          });
          pairs.forEach(([, to], i) => {
            column = renameInColumn(column, held[i], to) as Record<string, unknown>;
          });
          column = { ...column, name: own };
          nodes[id] = { ...node, params: { ...node.params, column } };
        }
        const edges = { ...s.graph.edges };
        const edgeOrder = [...s.graph.edgeOrder];
        for (const edge of fragment.edges) {
          if (edges[edge.id] || !nodes[edge.source] || !nodes[edge.target])
            continue;
          edges[edge.id] = edge;
          edgeOrder.push(edge.id);
        }
        // A pasted column reading one that was not copied with it reads the recipe's own: it gets
        // that wire too, so deleting the column it reads reaches it.
        const pasted = new Set(added.filter((id) => nodes[id].type === RECIPE_COLUMN));
        const wired = new Set(
          edgeOrder.map((id) => `${edges[id].source}>${edges[id].target}`),
        );
        for (const group of new Set([...pasted].map((id) => nodes[id].group))) {
          if (!group) continue;
          const columns = nodeOrder
            .map((id) => nodes[id])
            .filter((n) => n.type === RECIPE_COLUMN && n.group === group);
          for (const edge of referenceEdges(columns)) {
            if (!pasted.has(edge.target) || wired.has(`${edge.source}>${edge.target}`)) continue;
            const id = newId("e");
            edges[id] = { ...edge, id };
            edgeOrder.push(id);
          }
        }
        return withGraph(s, { nodes, nodeOrder, edges, edgeOrder });
      }),

    upsertGroup: (group) =>
      commit((s) =>
        withGraph(s, {
          groups: { ...s.graph.groups, [group.id]: group },
          groupOrder: s.graph.groups[group.id]
            ? s.graph.groupOrder
            : [...s.graph.groupOrder, group.id],
        }),
      ),

    resizeGroup: (group) =>
      commit((s) => {
        const groups = { ...s.graph.groups, [group.id]: group };
        const graph = { ...s.graph, groups };
        const nodes = { ...s.graph.nodes };
        // A frame folded into a closed card keeps its blocks and takes no others; a block hidden in
        // a closed card keeps its frame too, since its saved position no longer says where it sits.
        const folded = foldedGroups(s.meta.editor, s.graph);
        const hidden = new Set(
          readSuperCards(s.meta.editor, s.graph)
            .filter((card) => !card.open)
            .flatMap((card) => card.members),
        );
        for (const id of s.graph.nodeOrder) {
          const node = nodes[id];
          const before = node.group ?? null;
          if (hidden.has(id) || (before && folded.has(before))) continue;
          if (recipeOwned(node, s.graph.groups)) continue;
          const after = frameAt(graph, node.pos, folded);
          if (before === after) continue;
          if (before !== group.id && after !== group.id) continue;
          nodes[id] = { ...node, group: after };
        }
        return withGraph(s, { groups, nodes });
      }),

    groupNodes: (group, nodeIds) =>
      commit((s) => {
        const nodes = { ...s.graph.nodes };
        for (const id of nodeIds)
          if (nodes[id] && !recipeOwned(nodes[id], s.graph.groups))
            nodes[id] = { ...nodes[id], group: group.id };
        return withGraph(s, {
          nodes,
          groups: { ...s.graph.groups, [group.id]: group },
          groupOrder: s.graph.groups[group.id]
            ? s.graph.groupOrder
            : [...s.graph.groupOrder, group.id],
        });
      }),

    upsertNote: (note, coalesceKey) =>
      commit(
        (s) =>
          withGraph(s, {
            notes: { ...s.graph.notes, [note.id]: note },
            noteOrder: s.graph.notes[note.id]
              ? s.graph.noteOrder
              : [...s.graph.noteOrder, note.id],
          }),
        coalesceKey,
      ),

    patchGraph: (patch, coalesceKey) =>
      commit((s) => {
        const nodes = { ...s.graph.nodes };
        let nodeOrder = s.graph.nodeOrder;
        for (const node of patch.nodes ?? []) {
          if (!nodes[node.id]) nodeOrder = [...nodeOrder, node.id];
          nodes[node.id] = node;
        }
        const groups = { ...s.graph.groups };
        let groupOrder = s.graph.groupOrder;
        for (const group of patch.groups ?? []) {
          if (!groups[group.id]) groupOrder = [...groupOrder, group.id];
          groups[group.id] = group;
        }
        const drop = new Set(patch.removeEdges ?? []);
        const edges = { ...s.graph.edges };
        for (const id of drop) delete edges[id];
        let edgeOrder = drop.size
          ? s.graph.edgeOrder.filter((id) => !drop.has(id))
          : s.graph.edgeOrder;
        for (const edge of patch.addEdges ?? []) {
          if (!edges[edge.id]) edgeOrder = [...edgeOrder, edge.id];
          edges[edge.id] = edge;
        }
        return withGraph(s, {
          nodes,
          nodeOrder,
          groups,
          groupOrder,
          edges,
          edgeOrder,
        });
      }, coalesceKey),

    undo: () => {
      const s = get();
      const previous = s.past.at(-1);
      if (!previous) return;
      set({
        graph: previous.graph,
        meta: previous.meta,
        past: s.past.slice(0, -1),
        future: [{ graph: s.graph, meta: s.meta }, ...s.future].slice(
          0,
          HISTORY_LIMIT,
        ),
        version: s.version + 1,
        lastCoalesceKey: null,
      });
    },

    redo: () => {
      const s = get();
      const next = s.future[0];
      if (!next) return;
      set({
        graph: next.graph,
        meta: next.meta,
        past: [...s.past, { graph: s.graph, meta: s.meta }].slice(
          -HISTORY_LIMIT,
        ),
        future: s.future.slice(1),
        version: s.version + 1,
        lastCoalesceKey: null,
      });
    },
  };
});

/**
 * Drops reroutes and the wires touching removed nodes or removed segments. A removed reroute heals:
 * what fed it is wired straight to what it fed.
 */
function removeReroutes(
  graph: DocGraph,
  dropReroutes: Set<string>,
  dropsEdge: (id: string) => boolean,
  dropsNode: (id: string) => boolean,
): { patch: Partial<DocGraph>; healed: Set<string> } {
  if (!dropReroutes.size && !graph.wireOrder.length)
    return { patch: {}, healed: new Set() };
  let wires: Record<string, RerouteWire> = {};
  let wireOrder: string[] = [];
  for (const id of graph.wireOrder) {
    const w = graph.wires[id];
    if (dropsEdge(id) || dropsNode(w.source) || dropsNode(w.target)) continue;
    wires[id] = w;
    wireOrder.push(id);
  }
  const plain: FlowEdge[] = [];
  for (const rid of dropReroutes) {
    const into = wireOrder.map((id) => wires[id]).find((w) => w.target === rid);
    const next: Record<string, RerouteWire> = {};
    const nextOrder: string[] = [];
    for (const id of wireOrder) {
      const w = wires[id];
      if (w.target === rid) continue;
      if (w.source !== rid) {
        next[id] = w;
        nextOrder.push(id);
        continue;
      }
      if (!into) continue;
      const healed = {
        ...w,
        source: into.source,
        source_port: into.source_port,
      };
      if (graph.reroutes[healed.source] && !dropReroutes.has(healed.source)) {
        next[id] = healed;
        nextOrder.push(id);
      } else if (
        !graph.reroutes[healed.target] &&
        !graph.reroutes[healed.source]
      ) {
        // A wire's pinned style is reroute metadata; the healed edge is a plain FlowEdge.
        const { style: _style, ...edge } = healed as typeof healed & { style?: unknown };
        plain.push({ ...edge, kind: "data" });
      } else {
        next[id] = healed;
        nextOrder.push(id);
      }
    }
    wires = next;
    wireOrder = nextOrder;
  }
  const reroutes = { ...graph.reroutes };
  for (const id of dropReroutes) delete reroutes[id];
  const edges = { ...graph.edges };
  const edgeOrder = graph.edgeOrder.filter((id) => !dropsEdge(id));
  for (const id of graph.edgeOrder) if (dropsEdge(id)) delete edges[id];
  for (const edge of plain) {
    edges[edge.id] = edge;
    if (!edgeOrder.includes(edge.id)) edgeOrder.push(edge.id);
  }
  return {
    healed: new Set(plain.map((edge) => edge.id)),
    patch: {
      reroutes,
      rerouteOrder: graph.rerouteOrder.filter((id) => !dropReroutes.has(id)),
      wires,
      wireOrder,
      ...(plain.length || graph.wireOrder.length ? { edges, edgeOrder } : {}),
    },
  };
}

function frameAt(
  graph: DocGraph,
  pos: [number, number],
  skip: ReadonlySet<string> = new Set(),
): string | null {
  const [x, y] = [pos[0] + 130, pos[1] + 16];
  let best: { id: string; area: number } | null = null;
  for (const id of graph.groupOrder) {
    if (skip.has(id)) continue;
    const [gx, gy, gw, gh] = graph.groups[id].rect;
    if (x < gx || y < gy || x > gx + gw || y > gy + gh) continue;
    if (!best || gw * gh < best.area) best = { id, area: gw * gh };
  }
  return best?.id ?? null;
}

export function cloneFragment(
  graph: DocGraph,
  nodeIds: string[],
  delta: [number, number],
): GraphFragment {
  const idMap = new Map<string, string>();
  const nodes: FlowNode[] = [];
  const from: Record<string, string | null> = {};
  for (const id of nodeIds) {
    const node = graph.nodes[id];
    if (!node) continue;
    const nextId = newId("n");
    idMap.set(id, nextId);
    from[nextId] = node.group ?? null;
    nodes.push({
      ...node,
      id: nextId,
      params: JSON.parse(JSON.stringify(node.params)) as Record<
        string,
        unknown
      >,
      pos: [node.pos[0] + delta[0], node.pos[1] + delta[1]],
      group: null,
    });
  }
  const edges: FlowEdge[] = [];
  for (const edgeId of graph.edgeOrder) {
    const edge = graph.edges[edgeId];
    const source = idMap.get(edge.source);
    const target = idMap.get(edge.target);
    if (source && target)
      edges.push({ ...edge, id: newId("e"), source, target });
  }
  return { nodes, edges, from };
}
