// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { replaceRef } from "@/features/recipe-studio/utils/refs";
import {
  FLOW_FORMAT,
  FLOW_FORMAT_VERSION,
  type FlowDocument,
  type FlowEdge,
  type FlowGroup,
  type FlowNode,
  type FlowNote,
} from "../types";

/** recipe-columns.ts GENERATE_TYPE; kept here so the store imports nothing from the recipe editor. */
const RECIPE_DATASET = "recipe.generate";
/** recipe-columns.ts COLUMN_TYPE. */
const RECIPE_COLUMN = "recipe.column";

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
): FlowNode {
  const name = columnNameOf(node);
  if (node.type !== RECIPE_COLUMN || !group || !name) return node;
  const taken = new Set(
    order
      .filter((id) => nodes[id].type === RECIPE_COLUMN && nodes[id].group === group)
      .map((id) => columnNameOf(nodes[id])),
  );
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

/** Every string in a column's definition with ``from`` renamed to ``to`` where it is a reference. */
function renameInColumn(value: unknown, from: string, to: string): unknown {
  if (typeof value === "string") return replaceRef(value, from, to);
  if (Array.isArray(value)) return value.map((v) => renameInColumn(v, from, to));
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value).map(([k, v]) => [k, renameInColumn(v, from, to)]),
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
    },
    coalesceKey?: string,
  ) => void;
  connect: (edge: FlowEdge, replaceSingleInput: boolean) => void;
  insertFragment: (fragment: GraphFragment) => void;
  upsertGroup: (group: FlowGroup) => void;
  resizeGroup: (group: FlowGroup) => void;
  groupNodes: (group: FlowGroup, nodeIds: string[]) => void;
  upsertNote: (note: FlowNote, coalesceKey?: string) => void;
  /** One undo step that upserts nodes and groups and adds or drops edges (a recipe group edit). */
  patchGraph: (patch: GraphPatch, coalesceKey?: string) => void;
  undo: () => void;
  redo: () => void;
}

const HISTORY_LIMIT = 100;
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
  return {
    nodes,
    nodeOrder,
    edges,
    edgeOrder,
    groups,
    groupOrder,
    notes,
    noteOrder,
  };
}

export function metaFromDocument(doc: FlowDocument): DocMeta {
  const { nodes: _n, edges: _e, groups: _g, notes: _no, ...meta } = doc;
  return {
    ...emptyMeta(doc.id),
    ...meta,
    format: FLOW_FORMAT,
    requirements: meta.requirements ?? { models: [] },
  };
}

export function documentFromState(
  state: Pick<DocState, "meta" | "graph">,
): FlowDocument {
  const { meta, graph } = state;
  return {
    ...meta,
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
  const { editor: _editor, ...rest } = canonical;
  return JSON.stringify(rest);
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
            [node.id]: { ...node, group: node.group ?? frameAt(s.graph, node.pos) },
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
        for (const [key, pos] of Object.entries(positions)) {
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
        if (!nodes && !groups && !notes) return null;
        return withGraph(s, {
          ...(nodes ? { nodes } : {}),
          ...(groups ? { groups } : {}),
          ...(notes ? { notes } : {}),
        });
      }),

    removeElements: (ids, coalesceKey) =>
      commit((s) => {
        const dropNodes = new Set(ids.nodes ?? []);
        const dropEdges = new Set(ids.edges ?? []);
        const dropGroups = new Set(ids.groups ?? []);
        const dropNotes = new Set(ids.notes ?? []);
        for (const edgeId of s.graph.edgeOrder) {
          const edge = s.graph.edges[edgeId];
          if (dropNodes.has(edge.source) || dropNodes.has(edge.target))
            dropEdges.add(edgeId);
        }
        if (
          !dropNodes.size &&
          !dropEdges.size &&
          !dropGroups.size &&
          !dropNotes.size
        )
          return null;
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
        return withGraph(s, {
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
        });
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
        return withGraph(s, { edges, edgeOrder: [...order, edge.id] });
      }),

    insertFragment: (fragment) =>
      commit((s) => {
        const nodes = { ...s.graph.nodes };
        const nodeOrder = [...s.graph.nodeOrder];
        // A copied column renamed to stay unique is renamed in the other copies that read it too.
        const renames = new Map<string, [string, string][]>();
        const added: string[] = [];
        for (const node of fragment.nodes) {
          if (nodes[node.id]) continue;
          const group = frameAt(s.graph, node.pos);
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
          const unique = uniqueColumn(node, group, nodes, nodeOrder);
          const from = columnNameOf(node);
          const to = columnNameOf(unique);
          if (group && from && to && from !== to)
            renames.set(group, [...(renames.get(group) ?? []), [from, to]]);
          nodes[node.id] = { ...unique, group, ...(extra ? { disabled: true } : {}) };
          nodeOrder.push(node.id);
          added.push(node.id);
        }
        for (const id of added) {
          const node = nodes[id];
          const pairs = node.group ? renames.get(node.group) : undefined;
          if (node.type !== RECIPE_COLUMN || !pairs) continue;
          const own = columnNameOf(node);
          let column = (node.params as { column: Record<string, unknown> }).column;
          for (const [from, to] of pairs)
            column = { ...(renameInColumn(column, from, to) as Record<string, unknown>), name: own };
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
        for (const id of s.graph.nodeOrder) {
          const node = nodes[id];
          const before = node.group ?? null;
          const after = frameAt(graph, node.pos);
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
          if (nodes[id]) nodes[id] = { ...nodes[id], group: group.id };
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

function frameAt(graph: DocGraph, pos: [number, number]): string | null {
  const [x, y] = [pos[0] + 130, pos[1] + 16];
  let best: { id: string; area: number } | null = null;
  for (const id of graph.groupOrder) {
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
  for (const id of nodeIds) {
    const node = graph.nodes[id];
    if (!node) continue;
    const nextId = newId("n");
    idMap.set(id, nextId);
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
  return { nodes, edges };
}
