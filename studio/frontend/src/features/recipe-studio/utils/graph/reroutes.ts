// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Edge, XYPosition } from "@xyflow/react";
import { HANDLE_IDS } from "../handles";

// A reroute is a canvas-only waypoint. The recipe still has one edge from the column to the
// column that uses it; the edge lists the reroutes it passes through, in order, in `data.via`,
// and the canvas draws it as one segment per hop.

export const REROUTE_WIDTH = 28;
export const REROUTE_HEIGHT = 16;

const SEGMENT_MARK = "~";

export type RerouteSegmentData = {
  /** The recipe edge this segment is part of. */
  logicalId: string;
  /** The two columns the recipe edge joins, for highlighting the whole wire. */
  wireEnds: [string, string];
};

export function isRerouteId(id: string, reroutes: Record<string, XYPosition>): boolean {
  return Object.prototype.hasOwnProperty.call(reroutes, id);
}

export function getEdgeVia(edge: Edge): string[] {
  const via = (edge.data as { via?: unknown } | undefined)?.via;
  return Array.isArray(via) ? via.filter((id): id is string => typeof id === "string") : [];
}

export function withEdgeVia(edge: Edge, via: string[]): Edge {
  const { via: _old, ...data } = (edge.data ?? {}) as Record<string, unknown>;
  return { ...edge, data: via.length > 0 ? { ...data, via } : data };
}

/** The recipe edge id behind a drawn wire segment. */
export function logicalEdgeId(displayId: string): string {
  const mark = displayId.lastIndexOf(SEGMENT_MARK);
  return mark === -1 ? displayId : displayId.slice(0, mark);
}

function segmentIndex(displayId: string): number {
  const mark = displayId.lastIndexOf(SEGMENT_MARK);
  return mark === -1 ? 0 : Number(displayId.slice(mark + 1)) || 0;
}

/** Splits every edge that runs through reroutes into one drawn segment per hop. */
export function expandRerouteEdges(
  edges: Edge[],
  reroutes: Record<string, XYPosition>,
): Edge[] {
  return edges.flatMap((edge) => {
    const via = getEdgeVia(edge).filter((id) => isRerouteId(id, reroutes));
    if (via.length === 0) {
      return [edge];
    }
    const stops = [edge.source, ...via, edge.target];
    const data: RerouteSegmentData = {
      logicalId: edge.id,
      wireEnds: [edge.source, edge.target],
    };
    return stops.slice(1).map((target, index) => ({
      ...edge,
      id: `${edge.id}${SEGMENT_MARK}${index}`,
      source: stops[index],
      target,
      sourceHandle: index === 0 ? edge.sourceHandle : HANDLE_IDS.rerouteOut,
      targetHandle: index === via.length ? edge.targetHandle : HANDLE_IDS.rerouteIn,
      data: { ...(edge.data ?? {}), ...data },
    }));
  });
}

/** Threads a free reroute into the drawn segment `displayId`. */
export function insertRerouteIntoEdge(
  edges: Edge[],
  displayId: string,
  rerouteId: string,
): Edge[] {
  const id = logicalEdgeId(displayId);
  const index = segmentIndex(displayId);
  return edges.map((edge) => {
    if (edge.id !== id) {
      return edge;
    }
    const via = getEdgeVia(edge);
    return withEdgeVia(edge, [...via.slice(0, index), rerouteId, ...via.slice(index)]);
  });
}

/** Drops reroutes from every wire that ran through them; the wires go straight again. */
export function removeReroutesFromEdges(edges: Edge[], ids: Set<string>): Edge[] {
  return edges.map((edge) => {
    const via = getEdgeVia(edge);
    const next = via.filter((id) => !ids.has(id));
    return next.length === via.length ? edge : withEdgeVia(edge, next);
  });
}

export function isRerouteInUse(edges: Edge[], rerouteId: string): boolean {
  return edges.some((edge) => getEdgeVia(edge).includes(rerouteId));
}

/** Where a reroute's input comes from: the edge feeding it and the reroutes up to and including it. */
export function findRerouteUpstream(
  edges: Edge[],
  rerouteId: string,
): { edge: Edge; via: string[] } | null {
  for (const edge of edges) {
    const via = getEdgeVia(edge);
    const at = via.indexOf(rerouteId);
    if (at !== -1) {
      return { edge, via: via.slice(0, at + 1) };
    }
  }
  return null;
}
