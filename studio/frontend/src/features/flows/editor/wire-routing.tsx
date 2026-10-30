// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  BaseEdge,
  type EdgeProps,
  type InternalNode,
  getBezierPath,
  useStore,
  useStoreApi,
} from "@xyflow/react";
import { type CSSProperties, memo, useEffect } from "react";
import { create } from "zustand";
import { useCatalogStore } from "../stores/catalog-store";
import type { RfEdge, RfNode } from "./derive-rf";
import { portInk } from "./theme-colors";
import { WireDelete, WireHalo, useWireHover } from "./wire-delete";
import { isWireObstacle, routingKey } from "./wire-obstacles";
import { stretchToHandles, wireMidpoint, wiresToReroute } from "./wire-live";
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
// Matches the canvas's VISIBLE_ONLY_THRESHOLD.
const MEASURE_ALL_LIMIT = 300;

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

function applyRoutes(
  routes: WireRoute[],
  ms: number | null,
  partial = false,
): void {
  const previous = useWireRoutes.getState().routes;
  const next = partial
    ? new Map<string, WireRoute>(previous)
    : new Map<string, WireRoute>();
  let changed = !partial && routes.length !== previous.size;
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
    applyRoutes(
      [...result.routes.values()],
      performance.now() - started,
      request.partial,
    );
    return;
  }
  if (!worker) {
    worker = new Worker(new URL("./wire-router.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (event: MessageEvent<RouteResponse>) => {
      inflight = false;
      if (event.data.routes)
        applyRoutes(
          event.data.routes,
          event.data.ms ?? null,
          event.data.partial,
        );
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

// A drag sends at most one request per animation frame.
let frame = 0;
function pumpNextFrame(): void {
  if (frame || typeof requestAnimationFrame === "undefined") return pump();
  frame = requestAnimationFrame(() => {
    frame = 0;
    pump();
  });
}

/**
 * Routes the canvas wires off the main thread whenever nodes or edges change. Bus mode only.
 * While cards move (`moving`), only their wires and the ones sharing a lane with them re-route,
 * once per frame; the rest keep their routes. One full pass runs on release.
 */
export const WireRouterController = memo(function WireRouterController({
  scope,
  moving = null,
}: {
  scope: string;
  moving?: ReadonlySet<string> | null;
}) {
  const store = useStoreApi<RfNode, RfEdge>();
  const nodesKey = useStore((s) => routingKey(s.nodes));
  const edges = useStore((s) => s.edges);

  useEffect(() => {
    if (!edges) return;
    // Route once every card has its size, so the first pass does not depend on measure timing.
    // Past the canvas's render-visible-only size off-screen cards never measure: route anyway.
    const lookup = store.getState().nodeLookup;
    if (lookup.size <= MEASURE_ALL_LIMIT)
      for (const node of lookup.values())
        if (!node.hidden && !(node.measured.width && node.measured.height))
          return;
    if (moving?.size) {
      const live = wiresToReroute(store.getState().edges, moving);
      // A note or a frame on its own moves no wire.
      if (!live.length) return;
      pending = () => ({
        id: ++nextId,
        scope,
        hopRadius: HOP_RADIUS,
        partial: true,
        ...buildRequest(
          store.getState().nodeLookup,
          wiresToReroute(store.getState().edges, moving),
        ),
      });
      pumpNextFrame();
      return;
    }
    pending = () => ({
      id: ++nextId,
      scope,
      hopRadius: HOP_RADIUS,
      ...buildRequest(store.getState().nodeLookup, store.getState().edges),
    });
    pump();
  }, [nodesKey, edges, scope, store, moving]);

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
  deletable,
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
  const { hover, enter, leave } = useWireHover();
  const flow = data?.kind === "flow";
  const base: CSSProperties = flow
    ? {
        stroke: selected ? "var(--foreground)" : "var(--muted-foreground)",
        strokeWidth: selected ? 3 : 2.25,
        strokeDasharray: "8 5",
        opacity: selected ? 1 : 0.8,
      }
    : {
        stroke: portInk(color),
        strokeWidth: selected ? 2.5 : 1.75,
        opacity: selected || hover ? 1 : 0.75,
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
  // A blocked route is a straight line through cards: draw the curve instead.
  const exact =
    route && !route.blocked
      ? alignToHandles(route, sourceX, sourceY, targetX, targetY)
      : null;
  const points =
    exact ??
    (route && !route.blocked
      ? stretchToHandles(route.points, sourceX, sourceY, targetX, targetY)
      : null);
  const remove = (x: number, y: number) => (
    <WireDelete
      id={id}
      x={x}
      y={y}
      show={deletable !== false && (hover || !!selected)}
      onEnter={enter}
      onLeave={leave}
    />
  );
  if (!(route && points)) {
    // Until a route lands (or when none exists), a curve: a step path would cut through cards.
    const [path, labelX, labelY] = getBezierPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
    });
    return (
      // biome-ignore lint/a11y/noStaticElementInteractions: hover only reveals the × button
      <g onMouseEnter={enter} onMouseLeave={leave}>
        {selected && <WireHalo path={path} />}
        <BaseEdge id={id} path={path} style={style} />
        {remove(labelX, labelY)}
      </g>
    );
  }
  // Hops and junction dots belong to the settled route only.
  const path = buildWirePath(
    points,
    exact ? route.hops : [],
    HOP_RADIUS,
    CORNER_RADIUS,
  );
  const mid = wireMidpoint(points);
  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: hover only reveals the × button
    <g onMouseEnter={enter} onMouseLeave={leave}>
      {selected && <WireHalo path={path} />}
      <BaseEdge id={id} path={path} style={style} />
      {remove(mid.x, mid.y)}
      {(exact ? route.junctions : []).map((point) => (
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
    </g>
  );
}

export const BusEdge = memo(BusEdgeImpl);
