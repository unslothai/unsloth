// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { findPort, getSpec } from "../stores/catalog-store";
import { type DocGraph, newId, useDocStore } from "../stores/doc-store";
import {
  type PortRef,
  REROUTE_DIRS,
  REROUTE_IN,
  type RerouteDir,
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

export function rerouteSize(dir: RerouteDir = "right") {
  return dir === "right"
    ? REROUTE_SIZE
    : { width: REROUTE_SIZE.height, height: REROUTE_SIZE.width };
}

export function nextRerouteDir(dir: RerouteDir = "right"): RerouteDir {
  return REROUTE_DIRS[(REROUTE_DIRS.indexOf(dir) + 1) % REROUTE_DIRS.length];
}

/** The way a reroute cut into a wire should face: along the wire where it lands. */
export function dirAlong(tangent: { x: number; y: number }): RerouteDir {
  if (Math.abs(tangent.y) <= Math.abs(tangent.x) * 1.5) return "right";
  return tangent.y > 0 ? "down" : "up";
}

/** The other wire style from the one wire `id` draws with now (Shift while cutting it). */
export function flippedStyle(
  graph: DocGraph,
  id: string,
  current: "bus" | "curved",
): "bus" | "curved" {
  return (graph.wires[id]?.style ?? current) === "bus" ? "curved" : "bus";
}

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

/** Whether a new input `port` could hang off this reroute's output. */
export function rerouteCanFeed(
  graph: DocGraph,
  rerouteId: string,
  port: PortSpec,
): boolean {
  if (port.type === "flow") return false;
  const source = portSpec(graph, rerouteUpstream(graph, rerouteId), "out");
  return !source || typesCompatible(source, port);
}

/** Whether output `port` could feed this reroute: it must fit every input the reroute feeds. */
export function rerouteTakes(
  graph: DocGraph,
  rerouteId: string,
  port: PortSpec,
): boolean {
  if (port.type === "flow") return false;
  return rerouteDownstream(graph, rerouteId).every((ref) => {
    const input = portSpec(graph, ref, "in");
    return !input || typesCompatible(port, input);
  });
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
  // An empty chain takes any mix of inputs: some outputs are typed `any` (Get JSON's value, a
  // file) and feed them all. The source is checked against every one once it is wired in.
  const source = portSpec(graph, upstream, "out");
  const edges = Object.values(graph.edges);
  for (const ref of targets) {
    const port = portSpec(graph, ref, "in");
    if (!port) continue;
    if (source && !typesCompatible(source, port))
      return {
        ok: false,
        reason: `${source.type} cannot connect to ${port.type}`,
      };
    // A multi input keeps every wire, so the same output through a reroute would arrive twice.
    if (
      upstream &&
      port.multi &&
      edges.some(
        (e) =>
          e.source === upstream.node &&
          e.source_port === upstream.port &&
          e.target === ref.node &&
          e.target_port === ref.port,
      )
    )
      return { ok: false, reason: "Those are already connected" };
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

/** Adds a reroute centred on `centre`, cutting `splitId` if given. */
export function addReroute(
  centre: { x: number; y: number },
  opts: {
    splitId?: string;
    style?: RerouteWire["style"];
    dir?: RerouteDir;
  } = {},
): string {
  const id = newId("rr");
  const dir = opts.dir ?? "right";
  const size = rerouteSize(dir);
  const pos: [number, number] = [
    Math.round(centre.x - size.width / 2),
    Math.round(centre.y - size.height / 2),
  ];
  useDocStore
    .getState()
    .addReroute(
      { id, pos, ...(dir === "right" ? {} : { dir }) },
      opts.splitId,
      opts.style,
    );
  return id;
}

/** Turns reroutes to their next direction about their centres. */
export function turnReroutes(ids: string[]): boolean {
  const { graph, replaceReroutes } = useDocStore.getState();
  const turned = ids.flatMap((id) => {
    const reroute = graph.reroutes[id];
    if (!reroute) return [];
    const from = rerouteSize(reroute.dir);
    const dir = nextRerouteDir(reroute.dir);
    const to = rerouteSize(dir);
    const { dir: _dir, ...rest } = reroute;
    return [
      {
        ...rest,
        ...(dir === "right" ? {} : { dir }),
        pos: [
          reroute.pos[0] + (from.width - to.width) / 2,
          reroute.pos[1] + (from.height - to.height) / 2,
        ] as [number, number],
      },
    ];
  });
  if (turned.length) replaceReroutes(turned);
  return turned.length > 0;
}
