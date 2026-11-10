// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  BaseEdge,
  type EdgeProps,
  type InternalNode,
  Position,
  getBezierPath,
  useStore,
  useStoreApi,
} from "@xyflow/react";
import { type CSSProperties, memo, useEffect, useState } from "react";
import { create } from "zustand";
import { useCatalogStore } from "../stores/catalog-store";
import { useUiStore } from "../stores/ui-store";
import type { RfEdge, RfNode } from "./derive-rf";
import { portInk } from "./theme-colors";
import { WireDelete, WireHalo, useWireHover } from "./wire-delete";
import { routingVersion } from "./wire-obstacles";
import { stretchToHandles, wireMidpoint } from "./wire-live";
import type { RouteRequest, RouteResponse } from "./wire-router.worker";
import { buildWirePath, wireEmphasis } from "./wire-path";
import type { WirePoint, WireRoute } from "./wire-router";
import {
  type RouteResult,
  WireScheduler,
  requestRouting,
} from "./wire-schedule";

const HOP_RADIUS = 4;
const CORNER_RADIUS = 10;
const DIM_OPACITY = 0.3;
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

let worker: Worker | null = null;
let landed: ((result: RouteResult) => void) | null = null;

function route(request: RouteRequest, done: (result: RouteResult) => void): void {
  if (typeof Worker === "undefined") {
    // No worker (tests, old browsers): route here, loading the router only when needed.
    void import("./wire-router").then(({ routeWires }) => {
      const started = performance.now();
      const result = routeWires(
        request.rects,
        request.edges,
        { hopRadius: request.hopRadius },
        undefined,
        request.zones,
      );
      done({
        routes: [...result.routes.values()],
        ms: performance.now() - started,
        partial: request.partial,
      });
    });
    return;
  }
  if (!worker) {
    worker = new Worker(new URL("./wire-router.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (event: MessageEvent<RouteResponse>) => {
      const finish = landed;
      landed = null;
      finish?.({
        routes: event.data.routes,
        ms: event.data.ms ?? null,
        partial: event.data.partial,
      });
    };
  }
  landed = done;
  worker.postMessage(request);
}

const scheduler = new WireScheduler({
  route,
  nextFrame: (run) => {
    if (typeof requestAnimationFrame === "undefined") return false;
    requestAnimationFrame(run);
    return true;
  },
  apply: ({ routes, ms, partial }) =>
    applyRoutes(routes ?? [], ms ?? null, partial),
});

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
  const global = useUiStore((s) => s.wireStyle);
  // Sizes live in the lookup: a view-built node (a closed super card) never gets them back in s.nodes.
  const [versionOf] = useState(routingVersion);
  const nodesVersion = useStore((s) => versionOf(s.nodeLookup.values()));
  const edges = useStore((s) => s.edges);

  useEffect(() => {
    if (!edges) return;
    requestRouting(scheduler, () => store.getState(), scope, moving, global);
  }, [nodesVersion, edges, scope, store, moving, global]);

  useEffect(
    () => () => {
      scheduler.reset();
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
  sourceUpright = false,
  targetUpright = false,
): WirePoint[] | null {
  const points = route.points;
  if (points.length < 2) return null;
  const first = points[0];
  const last = points[points.length - 1];
  // The port's own axis may be off by a few pixels; across it, barely at all.
  const fits = (p: WirePoint, x: number, y: number, upright: boolean) =>
    upright
      ? Math.abs(x - p.x) <= ENDPOINT_TOLERANCE && Math.abs(y - p.y) <= 12
      : Math.abs(y - p.y) <= ENDPOINT_TOLERANCE && Math.abs(x - p.x) <= 12;
  if (
    !fits(first, sourceX, sourceY, sourceUpright) ||
    !fits(last, targetX, targetY, targetUpright)
  )
    return null;
  // Stretch the ends out to the handles, keeping the first and last legs straight.
  return [
    sourceUpright ? { x: first.x, y: sourceY } : { x: sourceX, y: first.y },
    ...points.slice(1, -1),
    targetUpright ? { x: last.x, y: targetY } : { x: targetX, y: last.y },
  ];
}

const upright = (position: Position) =>
  position === Position.Top || position === Position.Bottom;

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
      ? alignToHandles(
          route,
          sourceX,
          sourceY,
          targetX,
          targetY,
          upright(sourcePosition),
          upright(targetPosition),
        )
      : null;
  const points =
    exact ??
    (route &&
    !route.blocked &&
    !upright(sourcePosition) &&
    !upright(targetPosition)
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
