// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
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
  removeElements: (ids: {
    nodes?: string[];
    edges?: string[];
    groups?: string[];
    notes?: string[];
    reroutes?: string[];
  }) => void;
  connect: (edge: FlowEdge, replaceSingleInput: boolean) => void;
  /** Adds (or moves) a reroute and threads it into `splitId`, a plain edge or a wire segment. */
  addReroute: (reroute: FlowReroute, splitId?: string) => void;
  /** Connects a wire with a reroute on at least one end. */
  connectWire: (wire: RerouteWire, replaceSingleInput: boolean) => void;
  insertFragment: (fragment: GraphFragment) => void;
  upsertGroup: (group: FlowGroup) => void;
  resizeGroup: (group: FlowGroup) => void;
  groupNodes: (group: FlowGroup, nodeIds: string[]) => void;
  upsertNote: (note: FlowNote, coalesceKey?: string) => void;
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
  // The viewport is not an edit; reroutes are.
  return JSON.stringify({
    ...rest,
    reroutes: editor.reroutes,
    reroute_wires: editor.reroute_wires,
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

    removeElements: (ids) =>
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
      }),

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
        const dropped = s.graph.wireOrder.filter(
          (id) => s.graph.edges[id] && !edges[id],
        );
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

    addReroute: (reroute, splitId) =>
      commit((s) => {
        if (s.graph.nodes[reroute.id]) return null;
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
          // The far half keeps the id, so the plain edge the compiler sees keeps its id too.
          const near: RerouteWire = {
            id: newId("w"),
            source: cut.source,
            source_port: cut.source_port,
            target: reroute.id,
            target_port: REROUTE_IN,
          };
          const far: RerouteWire = {
            id: cut.id,
            source: reroute.id,
            source_port: REROUTE_OUT,
            target: cut.target,
            target_port: cut.target_port,
          };
          const wires = { ...graph.wires, [near.id]: near, [far.id]: far };
          const wireOrder = wire
            ? graph.wireOrder.flatMap((id) =>
                id === wire.id ? [near.id, id] : [id],
              )
            : [...graph.wireOrder, near.id, far.id];
          graph = { ...graph, wires, wireOrder };
          if (plain) {
            const edges = { ...graph.edges };
            delete edges[plain.id];
            graph = {
              ...graph,
              edges,
              edgeOrder: graph.edgeOrder.filter((id) => id !== plain.id),
            };
          }
          graph = syncRerouteEdges(graph, s.graph.wireOrder);
        }
        return { graph };
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
        return {
          graph: syncRerouteEdges(
            { ...g, wires, wireOrder, edges, edgeOrder },
            g.wireOrder,
          ),
        };
      }),

    insertFragment: (fragment) =>
      commit((s) => {
        const nodes = { ...s.graph.nodes };
        const nodeOrder = [...s.graph.nodeOrder];
        for (const node of fragment.nodes) {
          if (nodes[node.id]) continue;
          nodes[node.id] = { ...node, group: frameAt(s.graph, node.pos) };
          nodeOrder.push(node.id);
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
        plain.push({ ...healed, kind: "data" });
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
