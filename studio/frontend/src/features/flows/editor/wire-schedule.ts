// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { InternalNode } from "@xyflow/react";
import type { RfEdge, RfNode } from "./derive-rf";
import { sameValue } from "./same-value";
import { wiresToReroute } from "./wire-live";
import { isWireObstacle } from "./wire-obstacles";
import type { WireEdgeInput, WireRect, WireRoute, WireSide } from "./wire-router";
import type { RouteRequest } from "./wire-router.worker";

const HOP_RADIUS = 4;
const ZONE_MARGIN = 12;
// Matches the canvas's VISIBLE_ONLY_THRESHOLD.
const MEASURE_ALL_LIMIT = 300;
// A drag's partial pass sees only the cards this far around its wires; release routes everything.
const PARTIAL_WINDOW = 800;

type Lookup = Map<string, InternalNode<RfNode>>;
type Box = { x0: number; y0: number; x1: number; y1: number };

function handleBounds(
  node: InternalNode<RfNode> | undefined,
  handleId: string | null | undefined,
  kind: "source" | "target",
) {
  return node?.internals.handleBounds?.[kind]?.find(
    (handle) => handle.id === (handleId ?? null),
  );
}

function overlaps(rect: WireRect, box: Box): boolean {
  return (
    rect.x < box.x1 &&
    rect.x + rect.width > box.x0 &&
    rect.y < box.y1 &&
    rect.y + rect.height > box.y0
  );
}

export function buildRequest(
  lookup: Lookup,
  edges: RfEdge[],
  windowed = false,
): Pick<RouteRequest, "rects" | "zones" | "edges"> {
  let rects: WireRect[] = [];
  let zones: WireRect[] = [];
  for (const node of lookup.values()) {
    const width = node.measured.width;
    const height = node.measured.height;
    if (!(width && height) || node.hidden || !isWireObstacle(node.type))
      continue;
    const { x, y } = node.internals.positionAbsolute;
    if (node.type === "flowGroup") {
      zones.push({
        id: node.id,
        x: x - ZONE_MARGIN,
        y: y - ZONE_MARGIN,
        width: width + ZONE_MARGIN * 2,
        height: height + ZONE_MARGIN * 2,
      });
      continue;
    }
    rects.push({ id: node.id, x, y, width, height });
  }
  const wires: WireEdgeInput[] = [];
  for (const edge of edges) {
    if (edge.hidden) continue;
    const source = lookup.get(edge.source);
    const target = lookup.get(edge.target);
    const from = handleBounds(source, edge.sourceHandle, "source");
    const to = handleBounds(target, edge.targetHandle, "target");
    wires.push({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      sourceSide: (from?.position as WireSide | undefined) ?? "right",
      targetSide: (to?.position as WireSide | undefined) ?? "left",
      sourceOffset: from ? from.y + from.height / 2 : undefined,
      targetOffset: to ? to.y + to.height / 2 : undefined,
      net: `${edge.source}:${edge.sourceHandle ?? ""}`,
      sink: `${edge.target}:${edge.targetHandle ?? ""}`,
      ends: edge.data?.ends ?? [edge.source, edge.target],
    });
  }
  if (windowed) {
    const ends = new Set(wires.flatMap((wire) => [wire.source, wire.target]));
    const box = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
    for (const rect of rects)
      if (ends.has(rect.id)) {
        box.x0 = Math.min(box.x0, rect.x - PARTIAL_WINDOW);
        box.y0 = Math.min(box.y0, rect.y - PARTIAL_WINDOW);
        box.x1 = Math.max(box.x1, rect.x + rect.width + PARTIAL_WINDOW);
        box.y1 = Math.max(box.y1, rect.y + rect.height + PARTIAL_WINDOW);
      }
    rects = rects.filter((rect) => overlaps(rect, box));
    zones = zones.filter((zone) => overlaps(zone, box));
  }
  return { rects, zones, edges: wires };
}

/** The wires drawn as bus lines: the toolbar's style unless a wire is pinned to one. */
export function busEdges(edges: RfEdge[], global: "bus" | "curved"): RfEdge[] {
  return edges.filter((edge) => (edge.data?.wireStyle ?? global) === "bus");
}

export function sameRequest(a: RouteRequest, b: RouteRequest): boolean {
  return (
    a.scope === b.scope &&
    a.hopRadius === b.hopRadius &&
    !a.partial &&
    !b.partial &&
    sameValue(a.rects, b.rects) &&
    sameValue(a.zones, b.zones) &&
    sameValue(a.edges, b.edges)
  );
}

export type RouteResult = {
  routes?: WireRoute[];
  ms?: number | null;
  partial?: boolean;
};

export type SchedulerIO = {
  route: (request: RouteRequest, done: (result: RouteResult) => void) => void;
  nextFrame: (run: () => void) => boolean;
  apply: (result: RouteResult) => void;
};

// One pass in flight, latest graph queued; an unchanged full pass is skipped since selection re-derives edges.
export class WireScheduler {
  stats = { full: 0, partial: 0, skipped: 0 };
  private io: SchedulerIO;
  private pending: (() => RouteRequest) | null = null;
  private inflight = false;
  private frame = false;
  private lastFull: RouteRequest | null = null;

  constructor(io: SchedulerIO) {
    this.io = io;
  }

  schedule(build: () => RouteRequest, perFrame = false): void {
    this.pending = build;
    if (!perFrame) return this.pump();
    if (this.frame) return;
    this.frame = this.io.nextFrame(() => {
      this.frame = false;
      this.pump();
    });
    if (!this.frame) this.pump();
  }

  reset(): void {
    this.pending = null;
    this.lastFull = null;
  }

  private pump(): void {
    if (this.inflight || !this.pending) return;
    const request = this.pending();
    this.pending = null;
    if (request.partial) {
      this.lastFull = null;
      this.stats.partial += 1;
    } else if (this.lastFull && sameRequest(this.lastFull, request)) {
      this.stats.skipped += 1;
      return;
    } else {
      this.lastFull = request;
      this.stats.full += 1;
    }
    this.inflight = true;
    this.io.route(request, (result) => {
      this.inflight = false;
      if (result.routes) this.io.apply(result);
      else if (!request.partial && this.lastFull === request)
        this.lastFull = null;
      this.pump();
    });
  }
}

let nextId = 0;

export function requestRouting(
  scheduler: WireScheduler,
  getState: () => { nodeLookup: Lookup; edges: RfEdge[] },
  scope: string,
  moving: ReadonlySet<string> | null,
  global: "bus" | "curved",
): void {
  // Route once every card has its size, so the first pass does not depend on measure timing.
  // Past the canvas's render-visible-only size off-screen cards never measure: route anyway.
  const lookup = getState().nodeLookup;
  if (lookup.size <= MEASURE_ALL_LIMIT)
    for (const node of lookup.values())
      if (!node.hidden && !(node.measured.width && node.measured.height))
        return;
  if (moving?.size) {
    // A note or a frame on its own moves no wire.
    if (!wiresToReroute(busEdges(getState().edges, global), moving).length)
      return;
    scheduler.schedule(() => {
      const { nodeLookup, edges } = getState();
      return {
        id: ++nextId,
        scope,
        hopRadius: HOP_RADIUS,
        partial: true,
        ...buildRequest(
          nodeLookup,
          wiresToReroute(busEdges(edges, global), moving),
          true,
        ),
      };
    }, true);
    return;
  }
  scheduler.schedule(() => {
    const { nodeLookup, edges } = getState();
    return {
      id: ++nextId,
      scope,
      hopRadius: HOP_RADIUS,
      ...buildRequest(nodeLookup, busEdges(edges, global)),
    };
  });
}
