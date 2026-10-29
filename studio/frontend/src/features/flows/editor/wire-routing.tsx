// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  BaseEdge,
  type EdgeProps,
  type InternalNode,
  getSmoothStepPath,
  useStore,
  useStoreApi,
} from "@xyflow/react";
import { type CSSProperties, memo, useEffect } from "react";
import { create } from "zustand";
import { useCatalogStore } from "../stores/catalog-store";
import type { RfEdge, RfNode } from "./derive-rf";
import { isWireObstacle, routingKey } from "./wire-obstacles";
import type { RouteRequest, RouteResponse } from "./wire-router.worker";
import {
  type WireEdgeInput,
  type WirePoint,
  type WireRect,
  type WireRoute,
  buildWirePath,
  routeWires,
  wireEmphasis,
} from "./wire-router";

const HOP_RADIUS = 4;
const CORNER_RADIUS = 10;
const DIM_OPACITY = 0.3;
const ZONE_MARGIN = 12;
// Handles sit a few px outside the card; more than this means the route is from before a move.
const ENDPOINT_TOLERANCE = 2;

interface WireRouteState {
  routes: Map<string, WireRoute>;
  hovered: string | null;
  /** Worker time for the last routing pass, for the perf harness. */
  lastMs: number | null;
}

export const useWireRoutes = create<WireRouteState>(() => ({
  routes: new Map(),
  hovered: null,
  lastMs: null,
}));

export function setWireHover(nodeId: string | null): void {
  if (useWireRoutes.getState().hovered !== nodeId)
    useWireRoutes.setState({ hovered: nodeId });
}

function samePoints(a: WirePoint[], b: WirePoint[]): boolean {
  return (
    a.length === b.length && a.every((p, i) => p.x === b[i].x && p.y === b[i].y)
  );
}

function sameRoute(a: WireRoute, b: WireRoute): boolean {
  return (
    a.blocked === b.blocked &&
    samePoints(a.points, b.points) &&
    samePoints(a.hops, b.hops) &&
    samePoints(a.junctions, b.junctions) &&
    a.ends.length === b.ends.length &&
    a.ends.every((id, i) => id === b.ends[i])
  );
}

function applyRoutes(routes: WireRoute[], ms: number | null): void {
  const previous = useWireRoutes.getState().routes;
  const next = new Map<string, WireRoute>();
  let changed = routes.length !== previous.size;
  for (const route of routes) {
    const old = previous.get(route.id);
    if (old && sameRoute(old, route)) next.set(route.id, old);
    else {
      next.set(route.id, route);
      changed = true;
    }
  }
  useWireRoutes.setState(
    changed ? { routes: next, lastMs: ms } : { lastMs: ms },
  );
}

function handleCenter(
  node: InternalNode<RfNode> | undefined,
  handleId: string | null | undefined,
  kind: "source" | "target",
): number | undefined {
  const bounds = node?.internals.handleBounds?.[kind]?.find(
    (handle) => handle.id === (handleId ?? null),
  );
  return bounds ? bounds.y + bounds.height / 2 : undefined;
}

function buildRequest(
  lookup: Map<string, InternalNode<RfNode>>,
  edges: RfEdge[],
): Omit<RouteRequest, "id" | "scope" | "hopRadius"> {
  const rects: WireRect[] = [];
  const zones: WireRect[] = [];
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
    wires.push({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      sourceSide: "right",
      targetSide: "left",
      sourceOffset: handleCenter(
        lookup.get(edge.source),
        edge.sourceHandle,
        "source",
      ),
      targetOffset: handleCenter(
        lookup.get(edge.target),
        edge.targetHandle,
        "target",
      ),
      net: `${edge.source}:${edge.sourceHandle ?? ""}`,
      sink: `${edge.target}:${edge.targetHandle ?? ""}`,
      ends: edge.data?.ends ?? [edge.source, edge.target],
    });
  }
  return { rects, zones, edges: wires };
}

let worker: Worker | null = null;
let nextId = 0;
let inflight = false;
let pending: (() => RouteRequest) | null = null;

function send(request: RouteRequest): void {
  if (typeof Worker === "undefined") {
    const started = performance.now();
    const result = routeWires(
      request.rects,
      request.edges,
      { hopRadius: request.hopRadius },
      undefined,
      request.zones,
    );
    applyRoutes([...result.routes.values()], performance.now() - started);
    return;
  }
  if (!worker) {
    worker = new Worker(new URL("./wire-router.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (event: MessageEvent<RouteResponse>) => {
      inflight = false;
      if (event.data.routes)
        applyRoutes(event.data.routes, event.data.ms ?? null);
      pump();
    };
  }
  inflight = true;
  worker.postMessage(request);
}

// One request in flight; while it runs, only the latest graph is kept, and built when it is sent.
function pump(): void {
  if (inflight || !pending) return;
  const build = pending;
  pending = null;
  send(build());
}

/**
 * Routes the canvas wires off the main thread whenever nodes or edges change. Bus mode only.
 * While `paused` (a drag or resize) nothing is routed: the moved card's wires follow it as plain
 * steps and the rest hold still, then one pass runs on release.
 */
export const WireRouterController = memo(function WireRouterController({
  scope,
  paused = false,
}: {
  scope: string;
  paused?: boolean;
}) {
  const store = useStoreApi<RfNode, RfEdge>();
  const nodesKey = useStore((s) => routingKey(s.nodes));
  const edges = useStore((s) => s.edges);

  useEffect(() => {
    if (!edges || paused) return;
    pending = () => ({
      id: ++nextId,
      scope,
      hopRadius: HOP_RADIUS,
      ...buildRequest(store.getState().nodeLookup, store.getState().edges),
    });
    pump();
  }, [nodesKey, edges, scope, store, paused]);

  useEffect(
    () => () => {
      pending = null;
      useWireRoutes.setState({ routes: new Map(), hovered: null });
    },
    [],
  );
  return null;
});

function alignToHandles(
  route: WireRoute,
  sourceX: number,
  sourceY: number,
  targetX: number,
  targetY: number,
): WirePoint[] | null {
  const points = route.points;
  if (points.length < 2) return null;
  const first = points[0];
  const last = points[points.length - 1];
  if (
    Math.abs(sourceY - first.y) > ENDPOINT_TOLERANCE ||
    Math.abs(targetY - last.y) > ENDPOINT_TOLERANCE ||
    Math.abs(sourceX - first.x) > 12 ||
    Math.abs(targetX - last.x) > 12
  )
    return null;
  // Stretch the ends out to the handles, keeping the first and last legs straight.
  return [
    { x: sourceX, y: first.y },
    ...points.slice(1, -1),
    { x: targetX, y: last.y },
  ];
}

/** A routed wire: rounded bus corners, hops over crossings, a dot where a wire fans out. */
function BusEdgeImpl({
  id,
  source,
  target,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  selected,
  data,
}: EdgeProps<RfEdge>) {
  const color =
    useCatalogStore((s) => s.catalog?.port_colors[data?.portType ?? "any"]) ??
    "#94a3b8";
  const route = useWireRoutes((s) => s.routes.get(id));
  const ends = data?.ends;
  const emphasis = useWireRoutes((s) =>
    wireEmphasis(
      s.hovered ? new Set([s.hovered]) : new Set<string>(),
      ends ?? [source, target],
    ),
  );
  const flow = data?.kind === "flow";
  const base: CSSProperties = flow
    ? {
        stroke: selected ? "#ffffff" : "#e4e4e7",
        strokeWidth: selected ? 3 : 2.25,
        strokeDasharray: "8 5",
        opacity: selected ? 1 : 0.8,
      }
    : {
        stroke: color,
        strokeWidth: selected ? 3 : 1.75,
        opacity: selected ? 1 : 0.75,
      };
  const style: CSSProperties = {
    ...base,
    fill: "none",
    transition: "opacity 120ms ease",
    ...(emphasis === "focus"
      ? { opacity: 1, strokeWidth: (base.strokeWidth as number) + 0.6 }
      : emphasis === "dim"
        ? { opacity: DIM_OPACITY }
        : {}),
  };
  const points = route
    ? alignToHandles(route, sourceX, sourceY, targetX, targetY)
    : null;
  if (!(route && points)) {
    const [path] = getSmoothStepPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
      borderRadius: CORNER_RADIUS,
      offset: 20,
    });
    return <BaseEdge id={id} path={path} style={style} />;
  }
  return (
    <>
      <BaseEdge
        id={id}
        path={buildWirePath(points, route.hops, HOP_RADIUS, CORNER_RADIUS)}
        style={style}
      />
      {route.junctions.map((point) => (
        <circle
          key={`${point.x}:${point.y}`}
          cx={point.x}
          cy={point.y}
          r={3}
          fill={style.stroke as string}
          opacity={style.opacity}
          className="pointer-events-none"
        />
      ))}
    </>
  );
}

export const BusEdge = memo(BusEdgeImpl);
