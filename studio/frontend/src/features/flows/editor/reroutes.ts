// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { findPort, getSpec } from "../stores/catalog-store";
import { type DocGraph, newId, useDocStore } from "../stores/doc-store";
import {
  type PortRef,
  REROUTE_IN,
  REROUTE_OUT,
  type RerouteWire,
  rerouteDownstream,
  rerouteFeeds,
  rerouteUpstream,
  wireInto,
} from "../stores/reroute-graph";
import type { PortSpec } from "../types";
import {
  inHandle,
  outHandle,
  parseHandle,
  typesCompatible,
  wouldCreateCycle,
} from "./connection-rules";

export const REROUTE_TYPE = "reroute";
export const REROUTE_IN_HANDLE = "ri";
export const REROUTE_OUT_HANDLE = "ro";
export const REROUTE_SIZE = { width: 28, height: 16 };

const ANY: PortSpec = { id: "any", label: "Any", type: "any" } as PortSpec;

function portSpec(
  graph: DocGraph,
  ref: PortRef | null,
  side: "in" | "out",
): PortSpec | null {
  if (!ref) return null;
  const node = graph.nodes[ref.node];
  return (node && findPort(getSpec(node.type), side, ref.port)) ?? null;
}

/** The type a reroute carries: what is wired into it, else what it feeds, else any. */
export function reroutePort(graph: DocGraph, rerouteId: string): PortSpec {
  const upstream = portSpec(graph, rerouteUpstream(graph, rerouteId), "out");
  if (upstream) return upstream;
  const downstream = rerouteDownstream(graph, rerouteId)
    .map((ref) => portSpec(graph, ref, "in"))
    .find((port) => port && port.type !== "any");
  return downstream ?? ANY;
}

export function wireHandles(
  graph: DocGraph,
  wire: RerouteWire,
): { sourceHandle: string; targetHandle: string } {
  return {
    sourceHandle: graph.reroutes[wire.source]
      ? REROUTE_OUT_HANDLE
      : outHandle(wire.source_port),
    targetHandle: graph.reroutes[wire.target]
      ? REROUTE_IN_HANDLE
      : inHandle(wire.target_port),
  };
}

function chainRoot(graph: DocGraph, rerouteId: string): string {
  let cursor = rerouteId;
  const seen = new Set<string>();
  while (!seen.has(cursor)) {
    seen.add(cursor);
    const into = wireInto(graph, cursor);
    if (!into || !graph.reroutes[into.source]) break;
    cursor = into.source;
  }
  return cursor;
}

function typedBy(graph: DocGraph, rerouteId: string): PortSpec | null {
  const port = reroutePort(graph, rerouteId);
  return port.type === "any" ? null : port;
}

export interface RerouteRequest {
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
}

export function touchesReroute(
  graph: DocGraph,
  req: { source: string; target: string },
) {
  return !!graph.reroutes[req.source] || !!graph.reroutes[req.target];
}

/** Checks a connection with a reroute on either end; on success returns the wire to add. */
export function checkRerouteConnection(
  graph: DocGraph,
  req: RerouteRequest,
):
  | { ok: true; wire: RerouteWire; multi: boolean }
  | { ok: false; reason: string } {
  const fromReroute = !!graph.reroutes[req.source];
  const toReroute = !!graph.reroutes[req.target];
  if (req.source === req.target)
    return { ok: false, reason: "A reroute cannot feed itself" };
  let sourcePort: string;
  let upstream: PortRef | null;
  if (fromReroute) {
    if (req.sourceHandle !== REROUTE_OUT_HANDLE)
      return { ok: false, reason: "Connect an output to an input" };
    sourcePort = REROUTE_OUT;
    upstream = rerouteUpstream(graph, req.source);
  } else {
    const handle = parseHandle(req.sourceHandle);
    if (!handle || handle.side !== "out")
      return { ok: false, reason: "Connect an output to an input" };
    if (handle.kind !== "data")
      return { ok: false, reason: "Story paths do not go through reroutes" };
    if (!graph.nodes[req.source]) return { ok: false, reason: "Missing node" };
    sourcePort = handle.port;
    upstream = { node: req.source, port: handle.port };
    if (!portSpec(graph, upstream, "out"))
      return { ok: false, reason: "Unknown port" };
  }
  let targetPort: string;
  let targets: PortRef[];
  let multi = false;
  if (toReroute) {
    if (req.targetHandle !== REROUTE_IN_HANDLE)
      return { ok: false, reason: "Connect an output to an input" };
    if (fromReroute && rerouteFeeds(graph, req.target, req.source))
      return { ok: false, reason: "That would make a loop" };
    targetPort = REROUTE_IN;
    targets = rerouteDownstream(graph, req.target);
  } else {
    const handle = parseHandle(req.targetHandle);
    if (!handle || handle.side !== "in")
      return { ok: false, reason: "Connect an output to an input" };
    if (handle.kind !== "data")
      return { ok: false, reason: "Story paths do not go through reroutes" };
    if (!graph.nodes[req.target]) return { ok: false, reason: "Missing node" };
    targetPort = handle.port;
    targets = [{ node: req.target, port: handle.port }];
    const port = portSpec(graph, targets[0], "in");
    if (!port) return { ok: false, reason: "Unknown port" };
    multi = !!port.multi;
  }
  // An empty reroute chain takes its type from the inputs it already feeds.
  const source =
    portSpec(graph, upstream, "out") ??
    (fromReroute ? typedBy(graph, chainRoot(graph, req.source)) : null);
  const edges = Object.values(graph.edges);
  for (const ref of targets) {
    const port = portSpec(graph, ref, "in");
    if (!port) continue;
    if (source && !typesCompatible(source, port))
      return {
        ok: false,
        reason: `${source.type} cannot connect to ${port.type}`,
      };
    if (upstream && wouldCreateCycle(edges, upstream.node, ref.node, ref.port))
      return {
        ok: false,
        reason: "That would make a loop. Use For Each to repeat steps.",
      };
  }
  return {
    ok: true,
    multi,
    wire: {
      id: newId("w"),
      source: req.source,
      source_port: sourcePort,
      target: req.target,
      target_port: targetPort,
    },
  };
}

export function connectThroughReroute(req: RerouteRequest): string | null {
  const store = useDocStore.getState();
  const check = checkRerouteConnection(store.graph, req);
  if (!check.ok) return check.reason;
  store.connectWire(check.wire, !check.multi);
  return null;
}

export function addReroute(pos: [number, number], splitId?: string): string {
  const id = newId("rr");
  useDocStore.getState().addReroute({ id, pos }, splitId);
  return id;
}
