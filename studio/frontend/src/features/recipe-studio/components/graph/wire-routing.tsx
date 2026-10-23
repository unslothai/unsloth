// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  BaseEdge,
  getSmoothStepPath,
  type Edge,
  type EdgeProps,
  type InternalNode,
  type Node,
  useEdges,
  useNodes,
  useStoreApi,
} from "@xyflow/react";
import { type CSSProperties, type ReactElement, useEffect, useRef } from "react";
import { create } from "zustand";
import { getRecipeHandleSide } from "../../utils/handles";
import type { RerouteSegmentData } from "../../utils/graph/reroutes";
import {
  buildWirePath,
  findWireNear,
  routeWires,
  wireEmphasis,
  type WireEdgeInput,
  type WirePoint,
  type WireRect,
  type WireRoute,
  type WireRouterCache,
} from "../../utils/graph/wire-router";
import { readNodeHeight, readNodeWidth } from "../../utils/rf-node-dimensions";

const REROUTE_DEBOUNCE_MS = 120;
const HOP_RADIUS = 4;
// Handles sit a few px outside the card border; more than this sideways means the card moved.
const ENDPOINT_TOLERANCE = 2;
const DIM_OPACITY = 0.3;

type WireRouteState = {
  routes: Map<string, WireRoute>;
  /** Card under the pointer, then the selected cards: their wires stand out. */
  hovered: string | null;
  selected: ReadonlySet<string>;
  setRoutes: (routes: Map<string, WireRoute>) => void;
};

const useWireRouteStore = create<WireRouteState>((set) => ({
  routes: new Map(),
  hovered: null,
  selected: new Set(),
  setRoutes: (routes) => set({ routes }),
}));

export function setWireHover(nodeId: string | null): void {
  if (useWireRouteStore.getState().hovered !== nodeId) {
    useWireRouteStore.setState({ hovered: nodeId });
  }
}

/** The routed wire passing under a canvas point, skipping wires that touch `ignoreNodeId`. */
export function findWireAt(
  point: WirePoint,
  ignoreNodeId: string | null,
): { id: string; y: number | null } | null {
  const routes = [...useWireRouteStore.getState().routes.values()].filter(
    (route) => !(ignoreNodeId && route.ends.includes(ignoreNodeId)),
  );
  return findWireNear(routes, point, 10);
}

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

// Where along the card's side the handle sits, so each labelled port gets its own wire.
function handleOffset(
  node: InternalNode | undefined,
  handleId: string | null | undefined,
  kind: "source" | "target",
): number | undefined {
  const bounds = node?.internals.handleBounds?.[kind]?.find(
    (handle) => handle.id === (handleId ?? null),
  );
  return bounds ? bounds.y + bounds.height / 2 : undefined;
}

function toWireEdge(
  edge: Edge,
  lookup: Map<string, InternalNode>,
): WireEdgeInput | null {
  if (edge.hidden) {
    return null;
  }
  const sourceSide = getRecipeHandleSide(edge.sourceHandle, "source");
  const targetSide = getRecipeHandleSide(edge.targetHandle, "target");
  const wireEnds = (edge.data as Partial<RerouteSegmentData> | undefined)?.wireEnds;
  return {
    id: edge.id,
    source: edge.source,
    target: edge.target,
    sourceSide,
    targetSide,
    sourceOffset: handleOffset(lookup.get(edge.source), edge.sourceHandle, "source"),
    targetOffset: handleOffset(lookup.get(edge.target), edge.targetHandle, "target"),
    net: `${edge.source}:${edge.sourceHandle ?? sourceSide}`,
    sink: `${edge.target}:${edge.targetHandle ?? targetSide}`,
    ends: [edge.source, edge.target, ...(wireEnds ?? [])],
  };
}

/**
 * Routes every wire around the cards. Runs after a debounce while the graph changes and straight
 * away when a drag ends; during a drag the moved card's wires fall back to a plain step path.
 */
export function WireRouter(): null {
  const nodes = useNodes();
  const edges = useEdges();
  const store = useStoreApi();
  const setRoutes = useWireRouteStore((state) => state.setRoutes);
  const cacheRef = useRef<WireRouterCache | undefined>(undefined);
  const wasDraggingRef = useRef(false);

  const dragging = nodes.some((node) => node.dragging);
  const lookup = store.getState().nodeLookup;
  const rects = nodes.flatMap((node) => toRect(node) ?? []);
  const wires = edges.flatMap((edge) => toWireEdge(edge, lookup) ?? []);
  const geometryKey = JSON.stringify([rects, wires]);
  const inputRef = useRef({ rects, wires });
  inputRef.current = { rects, wires };
  const selectedKey = nodes
    .filter((node) => node.selected)
    .map((node) => node.id)
    .join("\n");

  useEffect(() => {
    useWireRouteStore.setState({
      selected: new Set(selectedKey ? selectedKey.split("\n") : []),
    });
  }, [selectedKey]);

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

  useEffect(
    () => () => {
      setRoutes(new Map());
      useWireRouteStore.setState({ hovered: null, selected: new Set() });
    },
    [setRoutes],
  );

  return null;
}

type WireEdgeProps = Pick<
  EdgeProps,
  | "id"
  | "source"
  | "target"
  | "sourceX"
  | "sourceY"
  | "targetX"
  | "targetY"
  | "sourcePosition"
  | "targetPosition"
  | "markerEnd"
  | "data"
> & { style: CSSProperties };

/** Draws the routed wire, or a plain step path while its cards are mid-drag. */
export function RoutedWire({
  id,
  source,
  target,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  data,
  style,
}: WireEdgeProps): ReactElement {
  const route = useWireRouteStore((state) => state.routes.get(id));
  const wireEnds = (data as Partial<RerouteSegmentData> | undefined)?.wireEnds;
  const emphasis = useWireRouteStore((state) =>
    wireEmphasis(
      state.hovered ? new Set([state.hovered]) : state.selected,
      [source, target, ...(wireEnds ?? [])],
    ),
  );
  const points = route ? alignToHandles(route, sourceX, sourceY, targetX, targetY) : null;
  const baseWidth = typeof style.strokeWidth === "number" ? style.strokeWidth : 2;
  const wireStyle: CSSProperties = {
    ...style,
    transition: "opacity 120ms ease, stroke 120ms ease",
    ...(emphasis === "focus"
      ? { stroke: "var(--primary)", strokeWidth: baseWidth + 0.6, opacity: 1 }
      : emphasis === "dim"
        ? { opacity: DIM_OPACITY }
        : {}),
  };

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
    return <BaseEdge id={id} path={path} markerEnd={markerEnd} style={wireStyle} />;
  }

  const path = buildWirePath(points, route.hops, HOP_RADIUS);
  const color = route.blocked ? "var(--destructive)" : wireStyle.stroke;
  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        markerEnd={markerEnd}
        style={{
          ...wireStyle,
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
          opacity={wireStyle.opacity}
          className="pointer-events-none"
          style={{ transition: wireStyle.transition }}
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
