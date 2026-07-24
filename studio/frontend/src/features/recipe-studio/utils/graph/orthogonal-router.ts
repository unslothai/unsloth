// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Compact obstacle-avoiding orthogonal edge router ("digital-logic" wiring).
 *
 * Strategy, cheapest-first so most edges never pay for search:
 *   1. Extend a short straight stub out of each handle (clean entry/exit).
 *   2. Fast path: try straight / L / Z connections between the stubs; if one
 *      clears every (inflated) node box, use it. Covers the common case.
 *   3. Only on collision, run A* over a Hanan grid (the x/y lines of the
 *      obstacle edges + endpoints). Cost = manhattan length + a turn penalty,
 *      which biases toward few-bend, right-angle "bus" routes.
 *   4. Emit a rounded-corner SVG path; fall back to smoothstep if fully boxed.
 *
 * Results are memoized on (endpoints + obstacle signature) so idle re-renders
 * are free and drags only recompute edges whose geometry actually moved.
 */

import { Position, getSmoothStepPath } from "@xyflow/react";

export type Rect = { x: number; y: number; width: number; height: number };
type Pt = { x: number; y: number };

/** Padding added around every node box so wires keep a visible gap. */
const CLEARANCE = 12;
/** Straight run out of each handle before the wire is allowed to turn. */
const STUB = 18;
/** Extra cost per 90° turn — higher = straighter, fewer-bend routes. */
const TURN_PENALTY = 24;
/** Corner rounding radius on emitted paths. */
const CORNER_RADIUS = 8;
/** Per-lane stub growth so parallel wires occupy distinct channels. */
const LANE_STEP = 9;
const LANE_COUNT = 4;
/** Grid points guard: bail to smoothstep past this (never hit in practice). */
const MAX_GRID_POINTS = 6000;
const EPS = 0.5;

export type OrthogonalRouteParams = {
  sourceX: number;
  sourceY: number;
  sourcePosition: Position;
  targetX: number;
  targetY: number;
  targetPosition: Position;
  obstacles: Rect[];
  /** Deterministic per-edge lane offset (see {@link laneOffsetFromId}). */
  laneOffset?: number;
};

/** Stable small offset derived from an edge id, to fan out parallel wires. */
export function laneOffsetFromId(id: string): number {
  let h = 0;
  for (let i = 0; i < id.length; i++) {
    h = (h * 31 + id.charCodeAt(i)) | 0;
  }
  return (Math.abs(h) % LANE_COUNT) * LANE_STEP;
}

function directionOf(position: Position): Pt {
  switch (position) {
    case Position.Left:
      return { x: -1, y: 0 };
    case Position.Right:
      return { x: 1, y: 0 };
    case Position.Top:
      return { x: 0, y: -1 };
    case Position.Bottom:
      return { x: 0, y: 1 };
    default:
      return { x: 1, y: 0 };
  }
}

function inflate(r: Rect, by: number): Rect {
  return {
    x: r.x - by,
    y: r.y - by,
    width: r.width + by * 2,
    height: r.height + by * 2,
  };
}

/** True when an axis-aligned segment passes through a rect's interior. */
function segmentHitsRect(a: Pt, b: Pt, r: Rect): boolean {
  const rx1 = r.x;
  const ry1 = r.y;
  const rx2 = r.x + r.width;
  const ry2 = r.y + r.height;
  if (Math.abs(a.y - b.y) < EPS) {
    const y = a.y;
    if (y <= ry1 + EPS || y >= ry2 - EPS) {
      return false;
    }
    const minX = Math.min(a.x, b.x);
    const maxX = Math.max(a.x, b.x);
    return Math.max(minX, rx1) < Math.min(maxX, rx2) - EPS;
  }
  const x = a.x;
  if (x <= rx1 + EPS || x >= rx2 - EPS) {
    return false;
  }
  const minY = Math.min(a.y, b.y);
  const maxY = Math.max(a.y, b.y);
  return Math.max(minY, ry1) < Math.min(maxY, ry2) - EPS;
}

function segmentHitsAny(a: Pt, b: Pt, rects: Rect[]): boolean {
  for (const r of rects) {
    if (segmentHitsRect(a, b, r)) {
      return true;
    }
  }
  return false;
}

/** Try straight / L / Z connections; returns a point list or null. */
function tryDirect(a: Pt, b: Pt, rects: Rect[]): Pt[] | null {
  if (Math.abs(a.x - b.x) < EPS || Math.abs(a.y - b.y) < EPS) {
    if (!segmentHitsAny(a, b, rects)) {
      return [a, b];
    }
  }
  const l1 = { x: b.x, y: a.y };
  if (!(segmentHitsAny(a, l1, rects) || segmentHitsAny(l1, b, rects))) {
    return [a, l1, b];
  }
  const l2 = { x: a.x, y: b.y };
  if (!(segmentHitsAny(a, l2, rects) || segmentHitsAny(l2, b, rects))) {
    return [a, l2, b];
  }
  const mx = Math.round((a.x + b.x) / 2);
  const z1 = { x: mx, y: a.y };
  const z2 = { x: mx, y: b.y };
  if (
    !(
      segmentHitsAny(a, z1, rects) ||
      segmentHitsAny(z1, z2, rects) ||
      segmentHitsAny(z2, b, rects)
    )
  ) {
    return [a, z1, z2, b];
  }
  const my = Math.round((a.y + b.y) / 2);
  const w1 = { x: a.x, y: my };
  const w2 = { x: b.x, y: my };
  if (
    !(
      segmentHitsAny(a, w1, rects) ||
      segmentHitsAny(w1, w2, rects) ||
      segmentHitsAny(w2, b, rects)
    )
  ) {
    return [a, w1, w2, b];
  }
  return null;
}

function uniqSortedRounded(values: number[]): number[] {
  const rounded = values.map((v) => Math.round(v));
  const sorted = Array.from(new Set(rounded)).sort((p, q) => p - q);
  return sorted;
}

/** Binary min-heap over (id, priority); lazy (allows duplicate pushes). */
class MinHeap {
  private ids: number[] = [];
  private ps: number[] = [];
  get size(): number {
    return this.ids.length;
  }
  push(id: number, p: number): void {
    this.ids.push(id);
    this.ps.push(p);
    let i = this.ids.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.ps[parent] <= this.ps[i]) {
        break;
      }
      this.swap(i, parent);
      i = parent;
    }
  }
  pop(): number {
    const top = this.ids[0];
    const lastId = this.ids.pop() as number;
    const lastP = this.ps.pop() as number;
    if (this.ids.length > 0) {
      this.ids[0] = lastId;
      this.ps[0] = lastP;
      let i = 0;
      const n = this.ids.length;
      for (;;) {
        const l = i * 2 + 1;
        const r = l + 1;
        let m = i;
        if (l < n && this.ps[l] < this.ps[m]) {
          m = l;
        }
        if (r < n && this.ps[r] < this.ps[m]) {
          m = r;
        }
        if (m === i) {
          break;
        }
        this.swap(i, m);
        i = m;
      }
    }
    return top;
  }
  private swap(a: number, b: number): void {
    const ti = this.ids[a];
    this.ids[a] = this.ids[b];
    this.ids[b] = ti;
    const tp = this.ps[a];
    this.ps[a] = this.ps[b];
    this.ps[b] = tp;
  }
}

/** A* over the Hanan grid between two points. Null if unreachable. */
// biome-ignore lint/complexity/noExcessiveCognitiveComplexity: A* search loop
function aStar(start: Pt, goal: Pt, rects: Rect[]): Pt[] | null {
  const xs = uniqSortedRounded([
    start.x,
    goal.x,
    ...rects.flatMap((r) => [r.x, r.x + r.width]),
  ]);
  const ys = uniqSortedRounded([
    start.y,
    goal.y,
    ...rects.flatMap((r) => [r.y, r.y + r.height]),
  ]);
  const w = xs.length;
  const h = ys.length;
  if (w * h > MAX_GRID_POINTS) {
    return null;
  }
  const xIndex = new Map(xs.map((v, i) => [v, i]));
  const yIndex = new Map(ys.map((v, i) => [v, i]));
  const sx = xIndex.get(Math.round(start.x));
  const sy = yIndex.get(Math.round(start.y));
  const gx = xIndex.get(Math.round(goal.x));
  const gy = yIndex.get(Math.round(goal.y));
  if (sx == null || sy == null || gx == null || gy == null) {
    return null;
  }

  const key = (ix: number, iy: number): number => iy * w + ix;
  const n = w * h;
  const g = new Float64Array(n).fill(Number.POSITIVE_INFINITY);
  const came = new Int32Array(n).fill(-1);
  const dir = new Int8Array(n).fill(-1); // 0 = horizontal, 1 = vertical
  const closed = new Uint8Array(n);
  const startId = key(sx, sy);
  const goalId = key(gx, gy);
  const heuristic = (ix: number, iy: number): number =>
    Math.abs(xs[ix] - goal.x) + Math.abs(ys[iy] - goal.y);

  g[startId] = 0;
  const heap = new MinHeap();
  heap.push(startId, heuristic(sx, sy));
  const steps: [number, number, number][] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 1],
    [0, -1, 1],
  ];

  while (heap.size > 0) {
    const cur = heap.pop();
    if (cur === goalId) {
      break;
    }
    if (closed[cur]) {
      continue;
    }
    closed[cur] = 1;
    const cix = cur % w;
    const ciy = (cur - cix) / w;
    const cp = { x: xs[cix], y: ys[ciy] };
    for (const [dix, diy, moveDir] of steps) {
      const nix = cix + dix;
      const niy = ciy + diy;
      if (nix < 0 || nix >= w || niy < 0 || niy >= h) {
        continue;
      }
      const nid = key(nix, niy);
      if (closed[nid]) {
        continue;
      }
      const np = { x: xs[nix], y: ys[niy] };
      if (segmentHitsAny(cp, np, rects)) {
        continue;
      }
      const len = Math.abs(np.x - cp.x) + Math.abs(np.y - cp.y);
      const turn = dir[cur] !== -1 && dir[cur] !== moveDir ? TURN_PENALTY : 0;
      const tentative = g[cur] + len + turn;
      if (tentative < g[nid]) {
        g[nid] = tentative;
        came[nid] = cur;
        dir[nid] = moveDir;
        heap.push(nid, tentative + heuristic(nix, niy));
      }
    }
  }

  if (goalId !== startId && came[goalId] === -1) {
    return null;
  }
  const points: Pt[] = [];
  let c = goalId;
  while (c !== -1) {
    const cix = c % w;
    const ciy = (c - cix) / w;
    points.push({ x: xs[cix], y: ys[ciy] });
    if (c === startId) {
      break;
    }
    c = came[c];
  }
  points.reverse();
  return points;
}

function dedupe(points: Pt[]): Pt[] {
  const out: Pt[] = [];
  for (const p of points) {
    const last = out[out.length - 1];
    if (!last || Math.abs(last.x - p.x) > EPS || Math.abs(last.y - p.y) > EPS) {
      out.push(p);
    }
  }
  return out;
}

/** Drop midpoints that lie on a straight run between their neighbours. */
function simplify(points: Pt[]): Pt[] {
  if (points.length <= 2) {
    return points;
  }
  const out: Pt[] = [points[0]];
  for (let i = 1; i < points.length - 1; i++) {
    const a = out[out.length - 1];
    const b = points[i];
    const c = points[i + 1];
    const collinear =
      (Math.abs(a.x - b.x) < EPS && Math.abs(b.x - c.x) < EPS) ||
      (Math.abs(a.y - b.y) < EPS && Math.abs(b.y - c.y) < EPS);
    if (!collinear) {
      out.push(b);
    }
  }
  out.push(points[points.length - 1]);
  return out;
}

function distance(a: Pt, b: Pt): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function towards(from: Pt, to: Pt, dist: number): Pt {
  const d = distance(from, to);
  if (d < EPS) {
    return { x: from.x, y: from.y };
  }
  const t = dist / d;
  return { x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t };
}

function toRoundedPath(points: Pt[], radius: number): string {
  if (points.length === 0) {
    return "";
  }
  if (points.length <= 2) {
    const a = points[0];
    const b = points[points.length - 1];
    return `M ${a.x},${a.y} L ${b.x},${b.y}`;
  }
  let d = `M ${points[0].x},${points[0].y}`;
  for (let i = 1; i < points.length - 1; i++) {
    const prev = points[i - 1];
    const corner = points[i];
    const next = points[i + 1];
    const r = Math.min(
      radius,
      distance(prev, corner) / 2,
      distance(corner, next) / 2,
    );
    const entry = towards(corner, prev, r);
    const exit = towards(corner, next, r);
    d += ` L ${entry.x},${entry.y} Q ${corner.x},${corner.y} ${exit.x},${exit.y}`;
  }
  const last = points[points.length - 1];
  d += ` L ${last.x},${last.y}`;
  return d;
}

const pathCache = new Map<string, string>();
const MAX_CACHE = 600;

function obstacleSignature(rects: Rect[]): string {
  // Sorted so equal layouts hash identically regardless of node order.
  return rects
    .map(
      (r) =>
        `${Math.round(r.x)},${Math.round(r.y)},${Math.round(r.width)},${Math.round(r.height)}`,
    )
    .sort()
    .join("|");
}

/**
 * Route an orthogonal, box-avoiding path and return an SVG `d` string.
 * Falls back to a smoothstep path when the target is fully enclosed.
 */
export function routeOrthogonalPath(params: OrthogonalRouteParams): string {
  const {
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    obstacles,
    laneOffset = 0,
  } = params;

  const cacheKey = `${Math.round(sourceX)},${Math.round(sourceY)},${sourcePosition}>${Math.round(targetX)},${Math.round(targetY)},${targetPosition}#${laneOffset}#${obstacleSignature(obstacles)}`;
  const cached = pathCache.get(cacheKey);
  if (cached !== undefined) {
    return cached;
  }

  const inflated = obstacles
    .filter((r) => r.width > 0 && r.height > 0)
    .map((r) => inflate(r, CLEARANCE));

  const sd = directionOf(sourcePosition);
  const td = directionOf(targetPosition);
  const stub = STUB + laneOffset;
  const source = { x: sourceX, y: sourceY };
  const target = { x: targetX, y: targetY };
  const sourceStub = { x: sourceX + sd.x * stub, y: sourceY + sd.y * stub };
  const targetStub = { x: targetX + td.x * stub, y: targetY + td.y * stub };

  let mid = tryDirect(sourceStub, targetStub, inflated);
  if (!mid) {
    mid = aStar(sourceStub, targetStub, inflated);
  }

  let result: string;
  if (mid) {
    const full = simplify(dedupe([source, ...mid, target]));
    result = toRoundedPath(full, CORNER_RADIUS);
  } else {
    const [fallback] = getSmoothStepPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
    });
    result = fallback;
  }

  if (pathCache.size > MAX_CACHE) {
    pathCache.clear();
  }
  pathCache.set(cacheKey, result);
  return result;
}
