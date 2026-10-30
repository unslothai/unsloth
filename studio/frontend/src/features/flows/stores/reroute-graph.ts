// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Reroutes are canvas-only waypoints. The document keeps one plain edge from the real output to the
// real input for every wire that runs through reroutes (so the compiler and engine never see them),
// and stores the reroutes and the wire segments around them under `editor`. In the editor the
// segments are the truth; the plain edges are derived from them.

import type { FlowEdge } from "../types";

export const REROUTE_IN = "in";
export const REROUTE_OUT = "out";

export interface FlowReroute {
  id: string;
  pos: [number, number];
}

/** A wire segment with a reroute on at least one end. The reroute side's port is "in" or "out". */
export interface RerouteWire {
  id: string;
  source: string;
  source_port: string;
  target: string;
  target_port: string;
}

export interface RerouteGraph {
  nodes: Record<string, unknown>;
  edges: Record<string, FlowEdge>;
  edgeOrder: string[];
  reroutes: Record<string, FlowReroute>;
  rerouteOrder: string[];
  wires: Record<string, RerouteWire>;
  wireOrder: string[];
}

export type PortRef = { node: string; port: string };

export function wireInto(
  graph: Pick<RerouteGraph, "wires" | "wireOrder">,
  rerouteId: string,
): RerouteWire | undefined {
  for (const id of graph.wireOrder) {
    const wire = graph.wires[id];
    if (wire.target === rerouteId) return wire;
  }
  return undefined;
}

/** The real output feeding a reroute, or null when nothing (or only a loop) is wired in. */
export function rerouteUpstream(
  graph: Pick<RerouteGraph, "reroutes" | "wires" | "wireOrder">,
  rerouteId: string,
): PortRef | null {
  let cursor = rerouteId;
  const seen = new Set<string>();
  while (graph.reroutes[cursor] && !seen.has(cursor)) {
    seen.add(cursor);
    const into = wireInto(graph, cursor);
    if (!into) return null;
    if (!graph.reroutes[into.source])
      return { node: into.source, port: into.source_port };
    cursor = into.source;
  }
  return null;
}

/** Every real input a reroute feeds, through any further reroutes. */
export function rerouteDownstream(
  graph: Pick<RerouteGraph, "reroutes" | "wires" | "wireOrder">,
  rerouteId: string,
): PortRef[] {
  const out: PortRef[] = [];
  const stack = [rerouteId];
  const seen = new Set<string>();
  while (stack.length) {
    const id = stack.pop() as string;
    if (seen.has(id)) continue;
    seen.add(id);
    for (const wireId of graph.wireOrder) {
      const wire = graph.wires[wireId];
      if (wire.source !== id) continue;
      if (graph.reroutes[wire.target]) stack.push(wire.target);
      else out.push({ node: wire.target, port: wire.target_port });
    }
  }
  return out;
}

/** True when `from` is `rerouteId` or sits downstream of it. */
export function rerouteFeeds(
  graph: Pick<RerouteGraph, "reroutes" | "wires" | "wireOrder">,
  rerouteId: string,
  from: string,
): boolean {
  let cursor: string | undefined = from;
  const seen = new Set<string>();
  while (cursor && graph.reroutes[cursor] && !seen.has(cursor)) {
    if (cursor === rerouteId) return true;
    seen.add(cursor);
    cursor = wireInto(graph, cursor)?.source;
  }
  return false;
}

/** Plain edges for every wire that reaches a real input from a real output through reroutes. */
export function derivedEdges(
  graph: Pick<RerouteGraph, "nodes" | "reroutes" | "wires" | "wireOrder">,
): FlowEdge[] {
  const out: FlowEdge[] = [];
  for (const id of graph.wireOrder) {
    const wire = graph.wires[id];
    if (graph.reroutes[wire.target] || !graph.nodes[wire.target]) continue;
    const from = rerouteUpstream(graph, wire.source);
    if (!from || !graph.nodes[from.node]) continue;
    out.push({
      id: wire.id,
      source: from.node,
      source_port: from.port,
      target: wire.target,
      target_port: wire.target_port,
      kind: "data",
    });
  }
  return out;
}

function sameEdge(a: FlowEdge | undefined, b: FlowEdge): boolean {
  return (
    !!a &&
    a.source === b.source &&
    a.source_port === b.source_port &&
    a.target === b.target &&
    a.target_port === b.target_port &&
    a.kind === b.kind
  );
}

/** Recomputes the derived plain edges after the wires changed; untouched edges keep their identity. */
export function syncRerouteEdges<G extends RerouteGraph>(
  graph: G,
  previousWireIds: Iterable<string> = [],
): G {
  const stale = new Set<string>(previousWireIds);
  for (const id of graph.wireOrder) stale.add(id);
  const derived = new Map(derivedEdges(graph).map((edge) => [edge.id, edge]));
  const edges: Record<string, FlowEdge> = {};
  const edgeOrder: string[] = [];
  const place = (edge: FlowEdge) => {
    const before = graph.edges[edge.id];
    edges[edge.id] = sameEdge(before, edge) ? before : edge;
    edgeOrder.push(edge.id);
  };
  // A derived edge keeps its place: multi inputs take their values in edge order.
  for (const id of graph.edgeOrder) {
    if (!stale.has(id)) {
      edges[id] = graph.edges[id];
      edgeOrder.push(id);
    } else if (derived.has(id) && !edges[id]) {
      place(derived.get(id) as FlowEdge);
    }
  }
  // Only a new wire lands at the end: a split keeps its far half's id, and healing reuses it.
  for (const edge of derived.values()) if (!edges[edge.id]) place(edge);
  const unchanged =
    edgeOrder.length === graph.edgeOrder.length &&
    edgeOrder.every(
      (id, i) => id === graph.edgeOrder[i] && edges[id] === graph.edges[id],
    );
  return unchanged ? graph : { ...graph, edges, edgeOrder };
}

function isPos(value: unknown): value is [number, number] {
  return (
    Array.isArray(value) &&
    value.length === 2 &&
    value.every((n) => typeof n === "number" && Number.isFinite(n))
  );
}

/** Reads `editor.reroutes` / `editor.reroute_wires`, dropping anything malformed or dangling. */
export function reroutesFromEditor(
  editor: Record<string, unknown> | undefined,
  nodes: Record<string, unknown>,
): {
  reroutes: Record<string, FlowReroute>;
  rerouteOrder: string[];
  wires: Record<string, RerouteWire>;
  wireOrder: string[];
} {
  const reroutes: Record<string, FlowReroute> = {};
  const rerouteOrder: string[] = [];
  const rawReroutes = Array.isArray(editor?.reroutes) ? editor.reroutes : [];
  for (const item of rawReroutes as Partial<FlowReroute>[]) {
    if (!item || typeof item.id !== "string" || !isPos(item.pos)) continue;
    if (reroutes[item.id] || nodes[item.id]) continue;
    reroutes[item.id] = { id: item.id, pos: [item.pos[0], item.pos[1]] };
    rerouteOrder.push(item.id);
  }
  const wires: Record<string, RerouteWire> = {};
  const wireOrder: string[] = [];
  const rawWires = Array.isArray(editor?.reroute_wires)
    ? editor.reroute_wires
    : [];
  const exists = (id: unknown) =>
    typeof id === "string" && (!!reroutes[id] || !!nodes[id]);
  for (const item of rawWires as Partial<RerouteWire>[]) {
    if (!item || typeof item.id !== "string" || wires[item.id]) continue;
    if (!exists(item.source) || !exists(item.target)) continue;
    if (!reroutes[item.source as string] && !reroutes[item.target as string])
      continue;
    if (
      typeof item.source_port !== "string" ||
      typeof item.target_port !== "string"
    )
      continue;
    wires[item.id] = {
      id: item.id,
      source: item.source as string,
      source_port: item.source_port,
      target: item.target as string,
      target_port: item.target_port,
    };
    wireOrder.push(item.id);
  }
  return { reroutes, rerouteOrder, wires, wireOrder };
}

export function reroutesToEditor(
  graph: Pick<
    RerouteGraph,
    "reroutes" | "rerouteOrder" | "wires" | "wireOrder"
  >,
): { reroutes?: FlowReroute[]; reroute_wires?: RerouteWire[] } {
  return {
    ...(graph.rerouteOrder.length
      ? { reroutes: graph.rerouteOrder.map((id) => graph.reroutes[id]) }
      : {}),
    ...(graph.wireOrder.length
      ? { reroute_wires: graph.wireOrder.map((id) => graph.wires[id]) }
      : {}),
  };
}
