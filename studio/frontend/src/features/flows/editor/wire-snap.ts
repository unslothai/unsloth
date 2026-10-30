// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type Point = { x: number; y: number };

/** The point on a path of length `length` nearest `p`: a coarse walk, then a fine one around the best sample. */
export function nearestOnPath(
  length: number,
  at: (distance: number) => Point,
  p: Point,
): Point {
  const dist = (q: Point) => (q.x - p.x) ** 2 + (q.y - p.y) ** 2;
  const coarse = Math.max(8, Math.ceil(length / 8));
  let best = 0;
  let bestD = Number.POSITIVE_INFINITY;
  for (let i = 0; i <= coarse; i++) {
    const d = dist(at((length * i) / coarse));
    if (d < bestD) [best, bestD] = [(length * i) / coarse, d];
  }
  const step = length / coarse;
  for (
    let s = Math.max(0, best - step);
    s <= Math.min(length, best + step);
    s += step / 16
  ) {
    const d = dist(at(s));
    if (d < bestD) [best, bestD] = [s, d];
  }
  return at(best);
}

/** The wire (edge id) under a screen point, skipping `ignore`. */
export function wireUnder(
  x: number,
  y: number,
  ignore?: string,
): string | null {
  for (const element of document.elementsFromPoint(x, y)) {
    const id = element.closest(".react-flow__edge")?.getAttribute("data-id");
    if (id && id !== ignore) return id;
  }
  return null;
}

/** Where on wire `id` a reroute dropped at flow point `p` sits, in flow coordinates. */
export function snapToWire(id: string, p: Point): Point | null {
  const path = document.querySelector<SVGPathElement>(
    `.react-flow__edge[data-id="${CSS.escape(id)}"] path.react-flow__edge-path`,
  );
  if (!path) return null;
  return nearestOnPath(
    path.getTotalLength(),
    (d) => path.getPointAtLength(d),
    p,
  );
}
