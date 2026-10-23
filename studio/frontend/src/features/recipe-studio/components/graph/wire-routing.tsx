// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  BaseEdge,
  getSmoothStepPath,
  type Edge,
  type EdgeProps,
  type Node,
  useEdges,
  useNodes,
} from "@xyflow/react";
import { type CSSProperties, type ReactElement, useEffect, useRef } from "react";
import { create } from "zustand";
import { getRecipeHandleSide } from "../../utils/handles";
import {
  buildWirePath,
  routeWires,
  type WireEdgeInput,
  type WireRect,
  type WireRoute,
  type WireRouterCache,
} from "../../utils/graph/wire-router";
import { readNodeHeight, readNodeWidth } from "../../utils/rf-node-dimensions";

const REROUTE_DEBOUNCE_MS = 120;
const HOP_RADIUS = 4;
// Handles sit a few px outside the card border; more than this sideways means the card moved.
const ENDPOINT_TOLERANCE = 2;

type WireRouteState = {
  routes: Map<string, WireRoute>;
  setRoutes: (routes: Map<string, WireRoute>) => void;
};

const useWireRouteStore = create<WireRouteState>((set) => ({
  routes: new Map(),
  setRoutes: (routes) => set({ routes }),
}));

function sameRoute(a: WireRoute | undefined, b: WireRoute): boolean {
  return (
    Boolean(a) &&
    a?.blocked === b.blocked &&
    JSON.stringify([a?.points, a?.hops, a?.junctions]) ===
      JSON.stringify([b.points, b.hops, b.junctions])
  );
}

function toRect(node: Node): WireRect | null {
  const width = node.measured?.width ?? readNodeWidth(node);
  const height = node.measured?.height ?? readNodeHeight(node);
  if (!(width && height) || node.hidden) {
    return null;
  }
  return { id: node.id, x: node.position.x, y: node.position.y, width, height };
}

function toWireEdge(edge: Edge): WireEdgeInput | null {
  if (edge.hidden) {
    return null;
  }
  const sourceSide = getRecipeHandleSide(edge.sourceHandle, "source");
  const targetSide = getRecipeHandleSide(edge.targetHandle, "target");
  return {
    id: edge.id,
    source: edge.source,
    target: edge.target,
    sourceSide,
    targetSide,
    net: `${edge.source}:${sourceSide}`,
  };
}

/**
 * Routes every wire around the cards. Runs after a debounce while the graph changes and straight
 * away when a drag ends; during a drag the moved card's wires fall back to a plain step path.
 */
export function WireRouter(): null {
  const nodes = useNodes();
  const edges = useEdges();
  const setRoutes = useWireRouteStore((state) => state.setRoutes);
  const cacheRef = useRef<WireRouterCache | undefined>(undefined);
  const wasDraggingRef = useRef(false);

  const dragging = nodes.some((node) => node.dragging);
  const rects = nodes.flatMap((node) => toRect(node) ?? []);
  const wires = edges.flatMap((edge) => toWireEdge(edge) ?? []);
  const geometryKey = JSON.stringify([rects, wires]);
  const inputRef = useRef({ rects, wires });
  inputRef.current = { rects, wires };

  useEffect(() => {
    const justDropped = wasDraggingRef.current && !dragging;
    wasDraggingRef.current = dragging;
    if (dragging) {
      return;
    }
    const timer = window.setTimeout(
      () => {
        const { rects: currentRects, wires: currentWires } = inputRef.current;
        const result = routeWires(
          currentRects,
          currentWires,
          { hopRadius: HOP_RADIUS },
          cacheRef.current,
        );
        cacheRef.current = result.cache;
        const previous = useWireRouteStore.getState().routes;
        const next = new Map<string, WireRoute>();
        for (const [id, route] of result.routes) {
          const old = previous.get(id);
          next.set(id, old && sameRoute(old, route) ? old : route);
        }
        setRoutes(next);
      },
      justDropped ? 0 : REROUTE_DEBOUNCE_MS,
    );
    return () => window.clearTimeout(timer);
  }, [dragging, geometryKey, setRoutes]);

  useEffect(() => () => setRoutes(new Map()), [setRoutes]);

  return null;
}

type WireEdgeProps = Pick<
  EdgeProps,
  | "id"
  | "sourceX"
  | "sourceY"
  | "targetX"
  | "targetY"
  | "sourcePosition"
  | "targetPosition"
  | "markerEnd"
> & { style: CSSProperties };

/** Draws the routed wire, or a plain step path while its cards are mid-drag. */
export function RoutedWire({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  style,
}: WireEdgeProps): ReactElement {
  const route = useWireRouteStore((state) => state.routes.get(id));
  const points = route ? alignToHandles(route, sourceX, sourceY, targetX, targetY) : null;

  if (!(route && points)) {
    const [path] = getSmoothStepPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
      borderRadius: 0,
      offset: 16,
    });
    return <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />;
  }

  const path = buildWirePath(points, route.hops, HOP_RADIUS);
  const color = route.blocked ? "var(--destructive)" : style.stroke;
  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        markerEnd={markerEnd}
        style={{
          ...style,
          stroke: color,
          strokeLinejoin: "miter",
          ...(route.blocked ? { strokeDasharray: "2 4" } : {}),
        }}
      />
      {route.junctions.map((point) => (
        <circle
          key={`${point.x}:${point.y}`}
          cx={point.x}
          cy={point.y}
          r={3.5}
          fill={typeof color === "string" ? color : "var(--muted-foreground)"}
          opacity={style.opacity}
          className="pointer-events-none"
        />
      ))}
    </>
  );
}

function alignToHandles(
  route: WireRoute,
  sourceX: number,
  sourceY: number,
  targetX: number,
  targetY: number,
): WireRoute["points"] | null {
  const points = route.points;
  if (points.length < 2) {
    return null;
  }
  const first = points[0];
  const last = points[points.length - 1];
  const sourceHorizontal = Math.abs(points[1].y - first.y) < 0.01;
  const targetHorizontal = Math.abs(points[points.length - 2].y - last.y) < 0.01;
  const sourceDrift = sourceHorizontal ? sourceY - first.y : sourceX - first.x;
  const targetDrift = targetHorizontal ? targetY - last.y : targetX - last.x;
  const sourceGap = sourceHorizontal ? sourceX - first.x : sourceY - first.y;
  const targetGap = targetHorizontal ? targetX - last.x : targetY - last.y;
  if (
    Math.abs(sourceDrift) > ENDPOINT_TOLERANCE ||
    Math.abs(targetDrift) > ENDPOINT_TOLERANCE ||
    Math.abs(sourceGap) > 12 ||
    Math.abs(targetGap) > 12
  ) {
    return null;
  }
  // Move the ends out to the handles along the wire, keeping every segment orthogonal.
  return [
    sourceHorizontal ? { x: sourceX, y: first.y } : { x: first.x, y: sourceY },
    ...points.slice(1, -1),
    targetHorizontal ? { x: targetX, y: last.y } : { x: last.x, y: targetY },
  ];
}
