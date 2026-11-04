// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RfEdge } from "./derive-rf";
import type { WirePoint } from "./wire-router";

const TOLERANCE = 2;

/**
 * The wires a drag re-routes live: those touching a moved card, plus the ones sharing an output
 * or an input with them, since those share a lane. Never every wire.
 */
export function wiresToReroute(
  edges: RfEdge[],
  moving: ReadonlySet<string>,
): RfEdge[] {
  const touches = (edge: RfEdge) =>
    (edge.data?.ends ?? [edge.source, edge.target]).some((id) =>
      moving.has(id),
    );
  const nets = new Set<string>();
  const sinks = new Set<string>();
  for (const edge of edges)
    if (touches(edge)) {
      nets.add(`${edge.source}:${edge.sourceHandle ?? ""}`);
      sinks.add(`${edge.target}:${edge.targetHandle ?? ""}`);
    }
  if (!nets.size) return [];
  return edges.filter(
    (edge) =>
      nets.has(`${edge.source}:${edge.sourceHandle ?? ""}`) ||
      sinks.has(`${edge.target}:${edge.targetHandle ?? ""}`),
  );
}

/** The point halfway along a routed wire, where its × button sits. */
export function wireMidpoint(points: WirePoint[]): WirePoint {
  let total = 0;
  for (let i = 1; i < points.length; i++)
    total += Math.hypot(
      points[i].x - points[i - 1].x,
      points[i].y - points[i - 1].y,
    );
  let left = total / 2;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1];
    const b = points[i];
    const len = Math.hypot(b.x - a.x, b.y - a.y);
    if (len >= left && len > 0) {
      const t = left / len;
      return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
    }
    left -= len;
  }
  return points[0] ?? { x: 0, y: 0 };
}

/**
 * The route a frame behind a moving card: its first and last legs carried to the handles, so the
 * wire follows the card until the new route lands. Null when the legs would fold back.
 */
export function stretchToHandles(
  points: WirePoint[],
  sourceX: number,
  sourceY: number,
  targetX: number,
  targetY: number,
): WirePoint[] | null {
  if (points.length < 4) return null;
  const out = points.map((p) => ({ ...p }));
  const n = out.length;
  const a = out[1];
  const b = out[n - 2];
  if (
    Math.abs(points[0].y - a.y) > TOLERANCE ||
    Math.abs(points[n - 1].y - b.y) > TOLERANCE ||
    sourceX > a.x - 4 ||
    targetX < b.x + 4
  )
    return null;
  out[0] = { x: sourceX, y: sourceY };
  out[1] = { x: a.x, y: sourceY };
  out[n - 2] = { x: b.x, y: targetY };
  out[n - 1] = { x: targetX, y: targetY };
  return out;
}
