// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Drawing a routed wire, kept apart from the router so the canvas loads without it.

import type { WirePoint } from "./wire-router";

const EPS = 0.01;

export function buildWirePath(
  points: WirePoint[],
  hops: WirePoint[] = [],
  radius = 4,
  corner = 0,
): string {
  if (points.length === 0) {
    return "";
  }
  const length = (a: WirePoint, b: WirePoint): number =>
    Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
  // Corner k rounds the bend at points[k]; it may take at most half of either leg.
  const corners = points.map((point, k) =>
    k === 0 || k === points.length - 1
      ? 0
      : Math.min(corner, length(points[k - 1], point) / 2, length(point, points[k + 1]) / 2),
  );
  const toward = (from: WirePoint, to: WirePoint, distance: number): WirePoint => ({
    x: from.x + Math.sign(to.x - from.x) * distance,
    y: from.y + Math.sign(to.y - from.y) * distance,
  });
  const parts = [`M ${points[0].x} ${points[0].y}`];
  for (let k = 0; k + 1 < points.length; k += 1) {
    const a = points[k];
    const b = points[k + 1];
    const from = toward(a, b, corners[k]);
    const to = toward(b, a, corners[k + 1]);
    if (Math.abs(a.y - b.y) < EPS && hops.length > 0) {
      const sign = Math.sign(b.x - a.x);
      const onSegment = hops
        .filter(
          (hop) =>
            Math.abs(hop.y - a.y) < EPS &&
            (hop.x - from.x) * sign > radius &&
            (to.x - hop.x) * sign > radius,
        )
        .sort((p, q) => (p.x - q.x) * sign);
      for (const hop of onSegment) {
        parts.push(`L ${hop.x - sign * radius} ${a.y}`);
        parts.push(
          `A ${radius} ${radius} 0 0 ${sign > 0 ? 1 : 0} ${hop.x + sign * radius} ${a.y}`,
        );
      }
    }
    parts.push(`L ${to.x} ${to.y}`);
    if (corners[k + 1] > EPS) {
      const next = toward(b, points[k + 2], corners[k + 1]);
      parts.push(`Q ${b.x} ${b.y} ${next.x} ${next.y}`);
    }
  }
  return parts.join(" ");
}

/** How a wire should be drawn given the cards in focus (hovered, else selected). */
export function wireEmphasis(
  focus: ReadonlySet<string>,
  ends: ReadonlyArray<string>,
): "none" | "focus" | "dim" {
  if (focus.size === 0) {
    return "none";
  }
  return ends.some((id) => focus.has(id)) ? "focus" : "dim";
}
