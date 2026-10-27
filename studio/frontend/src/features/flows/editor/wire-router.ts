// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Orthogonal, obstacle-avoiding wire router for the Flows canvas ("Bus" wires), ported from Recipe
// Studio. A* runs over a sparse grid built from the card edges (plus clearance) and the port stubs,
// with a cost for every bend, crossing and stretch shared with another wire. Wires that end up on the
// same line are then spread into lanes, crossings get a hop and fan-out points get a junction dot.
// Group frames are zones: only wires with an end inside a zone may enter it. Graphs too big for one
// grid route each wire on a grid of the cards around it instead.

export type WireSide = "left" | "top" | "right" | "bottom";

export type WirePoint = { x: number; y: number };

export type WireRect = {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

export type WireEdgeInput = {
  id: string;
  source: string;
  target: string;
  sourceSide: WireSide;
  targetSide: WireSide;
  /** Wires with the same net (one output handle) may share a path and fan out. */
  net: string;
  /** Wires into the same input handle may merge before it. Defaults to target + side. */
  sink?: string;
  /** Port position down the side from the card's top edge; the middle of the side if unset. */
  sourceOffset?: number;
  targetOffset?: number;
  /** Cards this wire belongs to, for highlighting. Defaults to source and target. */
  ends?: string[];
};

export type WireRoute = {
  id: string;
  points: WirePoint[];
  hops: WirePoint[];
  junctions: WirePoint[];
  /** No clean path existed; drawn as a plain orthogonal path instead. */
  blocked: boolean;
  ends: string[];
};

export type WireRouterOptions = {
  clearance?: number;
  laneGap?: number;
  bendPenalty?: number;
  crossingPenalty?: number;
  overlapPenalty?: number;
  clearancePenalty?: number;
  /** Per px for running alongside an unrelated wire a few px away. */
  nearPenalty?: number;
  /** Per px for running out on the ring around the whole graph. */
  ringPenalty?: number;
  hopRadius?: number;
  /** Above this many grid cells each wire is routed on its own local grid. */
  cellBudget?: number;
};

type RawRoute = { key: string; points: WirePoint[]; blocked: boolean };

export type WireRouterCache = {
  rects: Map<string, WireRect>;
  raw: Map<string, RawRoute>;
};

export type WireRouterResult = {
  routes: Map<string, WireRoute>;
  cache: WireRouterCache;
  /** Edges actually searched this call; the rest came from the cache. */
  rerouted: number;
};

const DIRS: ReadonlyArray<WirePoint> = [
  { x: 1, y: 0 },
  { x: 0, y: 1 },
  { x: -1, y: 0 },
  { x: 0, y: -1 },
];

const SIDE_DIR: Record<WireSide, number> = {
  right: 0,
  bottom: 1,
  left: 2,
  top: 3,
};

const EPS = 0.01;
const SHARED_DISCOUNT = 0.7;

export function wirePortPoint(rect: WireRect, side: WireSide, offset?: number): WirePoint {
  const along = offset ?? rect.height / 2;
  switch (side) {
    case "left":
      return { x: rect.x, y: rect.y + along };
    case "right":
      return { x: rect.x + rect.width, y: rect.y + along };
    case "top":
      return { x: rect.x + rect.width / 2, y: rect.y };
    default:
      return { x: rect.x + rect.width / 2, y: rect.y + rect.height };
  }
}

function onHalfPixel(point: WirePoint): WirePoint {
  return { x: Math.round(point.x * 2) / 2, y: Math.round(point.y * 2) / 2 };
}

function offsetPoint(point: WirePoint, dir: number, distance: number): WirePoint {
  return {
    x: point.x + DIRS[dir].x * distance,
    y: point.y + DIRS[dir].y * distance,
  };
}

function inflate(rect: WireRect, by: number): WireRect {
  return {
    id: rect.id,
    x: rect.x - by,
    y: rect.y - by,
    width: rect.width + by * 2,
    height: rect.height + by * 2,
  };
}

function sameRect(a: WireRect | undefined, b: WireRect | undefined): boolean {
  return Boolean(
    a &&
      b &&
      a.x === b.x &&
      a.y === b.y &&
      a.width === b.width &&
      a.height === b.height,
  );
}

function segmentHitsRect(a: WirePoint, b: WirePoint, rect: WireRect): boolean {
  const minX = Math.min(a.x, b.x);
  const maxX = Math.max(a.x, b.x);
  const minY = Math.min(a.y, b.y);
  const maxY = Math.max(a.y, b.y);
  return (
    maxX > rect.x + EPS &&
    minX < rect.x + rect.width - EPS &&
    maxY > rect.y + EPS &&
    minY < rect.y + rect.height - EPS
  );
}

function sortedUnique(values: number[]): number[] {
  const rounded = values.map((value) => Math.round(value * 2) / 2);
  rounded.sort((a, b) => a - b);
  const out: number[] = [];
  for (const value of rounded) {
    if (out.length === 0 || value - out[out.length - 1] > EPS) {
      out.push(value);
    }
  }
  return out;
}

// A line down the middle of every gap, so wires can use channel centres, not just card edges.
function withChannelMidlines(values: number[]): number[] {
  const out: number[] = [];
  for (let k = 0; k < values.length; k += 1) {
    if (k > 0 && values[k] - values[k - 1] > 24) {
      out.push(Math.round((values[k] + values[k - 1])) / 2);
    }
    out.push(values[k]);
  }
  return out;
}

function indexOf(values: number[], value: number): number {
  const target = Math.round(value * 2) / 2;
  let lo = 0;
  let hi = values.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (Math.abs(values[mid] - target) <= EPS) {
      return mid;
    }
    if (values[mid] < target) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return -1;
}

class MinHeap {
  private keys: number[] = [];
  private values: number[] = [];

  get size(): number {
    return this.keys.length;
  }

  push(key: number, value: number): void {
    const keys = this.keys;
    const values = this.values;
    let i = keys.length;
    keys.push(key);
    values.push(value);
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (keys[parent] <= key) {
        break;
      }
      keys[i] = keys[parent];
      values[i] = values[parent];
      i = parent;
    }
    keys[i] = key;
    values[i] = value;
  }

  pop(): number {
    const keys = this.keys;
    const values = this.values;
    const top = values[0];
    const lastKey = keys.pop() as number;
    const lastValue = values.pop() as number;
    if (keys.length > 0) {
      let i = 0;
      const n = keys.length;
      while (true) {
        const left = i * 2 + 1;
        if (left >= n) {
          break;
        }
        const right = left + 1;
        const child = right < n && keys[right] < keys[left] ? right : left;
        if (keys[child] >= lastKey) {
          break;
        }
        keys[i] = keys[child];
        values[i] = values[child];
        i = child;
      }
      keys[i] = lastKey;
      values[i] = lastValue;
    }
    return top;
  }
}

type Grid = {
  xs: number[];
  ys: number[];
  nx: number;
  ny: number;
  blockedPoint: Uint8Array;
  // Segment from point i to its right (h) or lower (v) neighbour passes through a card.
  blockedH: Uint8Array;
  blockedV: Uint8Array;
  // Segment runs through the clearance band around a card.
  softH: Uint8Array;
  softV: Uint8Array;
  // Innermost zone over a point or segment, -1 outside every zone.
  zonePoint: Int32Array;
  zoneH: Int32Array;
  zoneV: Int32Array;
  // Net index on each unit segment, -1 free, -2 shared by several nets.
  netH: Int32Array;
  netV: Int32Array;
  sinkH: Int32Array;
  sinkV: Int32Array;
  // Port stub points, kept clear for the wires that own them.
  reservedNet: Int32Array;
  reservedSink: Int32Array;
  /** Card bounds; grid lines outside them are the outer ring. */
  inner: { x0: number; x1: number; y0: number; y1: number };
  stamp: Int32Array;
  closed: Int32Array;
  gScore: Float64Array;
  cameFrom: Int32Array;
  search: number;
};

function markRects(
  rects: WireRect[],
  xs: number[],
  ys: number[],
  point: Uint8Array | null,
  horizontal: Uint8Array,
  vertical: Uint8Array,
): void {
  const ny = ys.length;
  for (const rect of rects) {
    const i0 = indexOf(xs, rect.x);
    const i1 = indexOf(xs, rect.x + rect.width);
    const j0 = indexOf(ys, rect.y);
    const j1 = indexOf(ys, rect.y + rect.height);
    for (let i = i0; i <= i1; i += 1) {
      for (let j = j0; j <= j1; j += 1) {
        const p = i * ny + j;
        const insideX = i > i0 && i < i1;
        const insideY = j > j0 && j < j1;
        if (point && insideX && insideY) {
          point[p] = 1;
        }
        if (insideY && i < i1) {
          horizontal[p] = 1;
        }
        if (insideX && j < j1) {
          vertical[p] = 1;
        }
      }
    }
  }
}

function markZones(
  zones: WireRect[],
  xs: number[],
  ys: number[],
  point: Int32Array,
  horizontal: Int32Array,
  vertical: Int32Array,
): void {
  const ny = ys.length;
  // Largest first, so a zone nested in another wins over it.
  const order = zones
    .map((zone, index) => ({ zone, index }))
    .sort((a, b) => b.zone.width * b.zone.height - a.zone.width * a.zone.height);
  for (const { zone, index } of order) {
    const i0 = indexOf(xs, zone.x);
    const i1 = indexOf(xs, zone.x + zone.width);
    const j0 = indexOf(ys, zone.y);
    const j1 = indexOf(ys, zone.y + zone.height);
    for (let i = i0; i <= i1; i += 1) {
      for (let j = j0; j <= j1; j += 1) {
        const p = i * ny + j;
        const insideX = i > i0 && i < i1;
        const insideY = j > j0 && j < j1;
        if (insideX && insideY) {
          point[p] = index;
        }
        if (insideY && i < i1) {
          horizontal[p] = index;
        }
        if (insideX && j < j1) {
          vertical[p] = index;
        }
      }
    }
  }
}

// Cards are hard obstacles; the clearance band around them only costs extra, so a port that
// sits close to another card can still get out.
function buildGrid(
  cards: WireRect[],
  margins: WireRect[],
  zones: WireRect[],
  extraXs: number[],
  extraYs: number[],
  ring: number,
): Grid {
  const rawXs = [...extraXs];
  const rawYs = [...extraYs];
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const rect of [...cards, ...margins, ...zones]) {
    rawXs.push(rect.x, rect.x + rect.width);
    rawYs.push(rect.y, rect.y + rect.height);
    minX = Math.min(minX, rect.x);
    minY = Math.min(minY, rect.y);
    maxX = Math.max(maxX, rect.x + rect.width);
    maxY = Math.max(maxY, rect.y + rect.height);
  }
  for (const x of extraXs) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
  }
  for (const y of extraYs) {
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  // Outer ring so wires that run backwards can go around the whole graph.
  rawXs.push(minX - ring, maxX + ring);
  rawYs.push(minY - ring, maxY + ring);

  const xs = withChannelMidlines(sortedUnique(rawXs));
  const ys = withChannelMidlines(sortedUnique(rawYs));
  const nx = xs.length;
  const ny = ys.length;
  const size = nx * ny;
  const blockedPoint = new Uint8Array(size);
  const blockedH = new Uint8Array(size);
  const blockedV = new Uint8Array(size);
  const softH = new Uint8Array(size);
  const softV = new Uint8Array(size);
  markRects(cards, xs, ys, blockedPoint, blockedH, blockedV);
  markRects(margins, xs, ys, null, softH, softV);
  const zonePoint = new Int32Array(size).fill(-1);
  const zoneH = new Int32Array(size).fill(-1);
  const zoneV = new Int32Array(size).fill(-1);
  markZones(zones, xs, ys, zonePoint, zoneH, zoneV);

  const netH = new Int32Array(size).fill(-1);
  const netV = new Int32Array(size).fill(-1);
  return {
    xs,
    ys,
    nx,
    ny,
    blockedPoint,
    blockedH,
    blockedV,
    softH,
    softV,
    zonePoint,
    zoneH,
    zoneV,
    netH,
    netV,
    sinkH: new Int32Array(size).fill(-1),
    sinkV: new Int32Array(size).fill(-1),
    reservedNet: new Int32Array(size).fill(-1),
    reservedSink: new Int32Array(size).fill(-1),
    inner: { x0: minX, x1: maxX, y0: minY, y1: maxY },
    stamp: new Int32Array(size * 4),
    closed: new Int32Array(size * 4),
    gScore: new Float64Array(size * 4),
    cameFrom: new Int32Array(size * 4),
    search: 0,
  };
}

function claim(values: Int32Array, p: number, id: number): void {
  const current = values[p];
  values[p] = current === -1 || current === id ? id : -2;
}

function markNet(grid: Grid, a: number, b: number, owner: Owner): void {
  const { ny } = grid;
  const ai = Math.floor(a / ny);
  const aj = a % ny;
  const bi = Math.floor(b / ny);
  const bj = b % ny;
  if (aj === bj) {
    const j = aj;
    for (let i = Math.min(ai, bi); i < Math.max(ai, bi); i += 1) {
      const p = i * ny + j;
      claim(grid.netH, p, owner.net);
      claim(grid.sinkH, p, owner.sink);
    }
    return;
  }
  const i = ai;
  for (let j = Math.min(aj, bj); j < Math.max(aj, bj); j += 1) {
    const p = i * ny + j;
    claim(grid.netV, p, owner.net);
    claim(grid.sinkV, p, owner.sink);
  }
}

function markPath(grid: Grid, points: WirePoint[], owner: Owner): void {
  for (let k = 0; k + 1 < points.length; k += 1) {
    const ai = indexOf(grid.xs, points[k].x);
    const aj = indexOf(grid.ys, points[k].y);
    const bi = indexOf(grid.xs, points[k + 1].x);
    const bj = indexOf(grid.ys, points[k + 1].y);
    if (ai < 0 || aj < 0 || bi < 0 || bj < 0 || (ai !== bi && aj !== bj)) {
      continue;
    }
    markNet(grid, ai * grid.ny + aj, bi * grid.ny + bj, owner);
  }
}

/** Source net (fan-out) and sink (fan-in) of a wire; wires sharing either may share a path. */
type Owner = { net: number; sink: number };

function sharesWith(net: number, sink: number, owner: Owner): boolean {
  return net === owner.net || sink === owner.sink;
}

function usedByOther(net: number, sink: number, owner: Owner): boolean {
  return net !== -1 && !sharesWith(net, sink, owner);
}

type SearchInput = {
  start: number;
  startDir: number;
  goal: number;
  goalDir: number;
  owner: Owner;
  window: { i0: number; i1: number; j0: number; j1: number };
  /** Cards stacked over this wire's own ports; it may cross them since it starts under them. */
  exempt: WireRect[];
  /** Zones this wire has an end in; every other zone is a wall. */
  zones: number[];
};

function inExempt(exempt: WireRect[], x: number, y: number): boolean {
  return exempt.some(
    (rect) =>
      x >= rect.x - EPS &&
      x <= rect.x + rect.width + EPS &&
      y >= rect.y - EPS &&
      y <= rect.y + rect.height + EPS,
  );
}

const NEAR_DISTANCE = 16;

// An unrelated wire on a neighbouring grid line, close enough to read as one bundle with this one.
function runsNearOther(
  grid: Grid,
  horizontal: boolean,
  i: number,
  j: number,
  owner: Owner,
): boolean {
  const { xs, ys, nx, ny } = grid;
  for (const step of [-1, 1]) {
    if (horizontal) {
      const nj = j + step;
      if (nj >= 0 && nj < ny && Math.abs(ys[nj] - ys[j]) <= NEAR_DISTANCE) {
        const q = i * ny + nj;
        if (usedByOther(grid.netH[q], grid.sinkH[q], owner)) {
          return true;
        }
      }
    } else {
      const ni = i + step;
      if (ni >= 0 && ni < nx && Math.abs(xs[ni] - xs[i]) <= NEAR_DISTANCE) {
        const q = ni * ny + j;
        if (usedByOther(grid.netV[q], grid.sinkV[q], owner)) {
          return true;
        }
      }
    }
  }
  return false;
}

function searchPath(
  grid: Grid,
  input: SearchInput,
  costs: Required<WireRouterOptions>,
): number[] | null {
  const { xs, ys, ny } = grid;
  const { start, startDir, goal, goalDir, owner, window, exempt, zones } = input;
  const walled = (zone: number): boolean => zone !== -1 && !zones.includes(zone);
  grid.search += 1;
  const search = grid.search;
  const goalX = xs[Math.floor(goal / ny)];
  const goalY = ys[goal % ny];
  const heuristic = (p: number): number => {
    const x = xs[Math.floor(p / ny)];
    const y = ys[p % ny];
    const dx = Math.abs(x - goalX);
    const dy = Math.abs(y - goalY);
    // Scaled by the shared-trunk discount so the estimate never overshoots.
    return (dx + dy) * SHARED_DISCOUNT + (dx > EPS && dy > EPS ? costs.bendPenalty : 0);
  };

  const heap = new MinHeap();
  const startState = start * 4 + startDir;
  grid.stamp[startState] = search;
  grid.gScore[startState] = 0;
  grid.cameFrom[startState] = -1;
  heap.push(heuristic(start), startState);

  while (heap.size > 0) {
    const state = heap.pop();
    const p = state >> 2;
    const dir = state & 3;
    const g = grid.gScore[state];
    if (p === goal) {
      return unwind(grid, state);
    }
    if (grid.closed[state] === search) {
      continue;
    }
    grid.closed[state] = search;

    const i = Math.floor(p / ny);
    const j = p % ny;
    for (let nd = 0; nd < 4; nd += 1) {
      if (nd === (dir + 2) % 4) {
        continue;
      }
      const ni = i + DIRS[nd].x;
      const nj = j + DIRS[nd].y;
      if (ni < window.i0 || ni > window.i1 || nj < window.j0 || nj > window.j1) {
        continue;
      }
      const q = ni * ny + nj;
      if (grid.blockedPoint[q] && q !== goal && !inExempt(exempt, xs[ni], ys[nj])) {
        continue;
      }
      if (walled(grid.zonePoint[q])) {
        continue;
      }
      const midX = (xs[ni] + xs[i]) / 2;
      const midY = (ys[nj] + ys[j]) / 2;
      let segNet: number;
      let segSink: number;
      let soft: boolean;
      if (nd === 0 || nd === 2) {
        const left = nd === 0 ? p : q;
        if ((grid.blockedH[left] && !inExempt(exempt, midX, midY)) || walled(grid.zoneH[left])) {
          continue;
        }
        segNet = grid.netH[left];
        segSink = grid.sinkH[left];
        soft = grid.softH[left] === 1;
      } else {
        const upper = nd === 1 ? p : q;
        if ((grid.blockedV[upper] && !inExempt(exempt, midX, midY)) || walled(grid.zoneV[upper])) {
          continue;
        }
        segNet = grid.netV[upper];
        segSink = grid.sinkV[upper];
        soft = grid.softV[upper] === 1;
      }
      const length = Math.abs(xs[ni] - xs[i]) + Math.abs(ys[nj] - ys[j]);
      let cost = soft ? length * 2 + costs.clearancePenalty : length;
      const horizontal = nd === 0 || nd === 2;
      const { inner } = grid;
      const onRing = horizontal
        ? ys[j] < inner.y0 || ys[j] > inner.y1
        : xs[i] < inner.x0 || xs[i] > inner.x1;
      if (onRing) {
        cost += length * costs.ringPenalty;
      }
      if (runsNearOther(grid, horizontal, Math.min(i, ni), Math.min(j, nj), owner)) {
        cost += length * costs.nearPenalty;
      }
      if (segNet !== -1 && sharesWith(segNet, segSink, owner)) {
        cost *= SHARED_DISCOUNT;
      } else if (usedByOther(segNet, segSink, owner)) {
        cost += costs.overlapPenalty + length * 2;
      }
      if (nd !== dir) {
        cost += costs.bendPenalty;
      }
      if (
        q !== goal &&
        ((grid.reservedNet[q] !== -1 && grid.reservedNet[q] !== owner.net) ||
          (grid.reservedSink[q] !== -1 && grid.reservedSink[q] !== owner.sink))
      ) {
        cost += costs.overlapPenalty;
      }
      if (q === goal && nd !== goalDir) {
        // Arriving along the target side instead of into it costs one more bend.
        cost += costs.bendPenalty;
      }
      if (q !== goal) {
        // A wire of another net running across q means this step crosses it.
        const crossing =
          nd === 0 || nd === 2
            ? usedByOther(grid.netV[q], grid.sinkV[q], owner) ||
              (nj > 0 && usedByOther(grid.netV[q - 1], grid.sinkV[q - 1], owner))
            : usedByOther(grid.netH[q], grid.sinkH[q], owner) ||
              (ni > 0 && usedByOther(grid.netH[q - ny], grid.sinkH[q - ny], owner));
        if (crossing) {
          cost += costs.crossingPenalty;
        }
      }
      const nextState = q * 4 + nd;
      const nextG = g + cost;
      if (grid.stamp[nextState] === search && grid.gScore[nextState] <= nextG) {
        continue;
      }
      grid.stamp[nextState] = search;
      grid.gScore[nextState] = nextG;
      grid.cameFrom[nextState] = state;
      heap.push(nextG + heuristic(q), nextState);
    }
  }
  return null;
}

function unwind(grid: Grid, state: number): number[] {
  const out: number[] = [];
  let current = state;
  let guard = 0;
  while (current >= 0 && guard < 100_000) {
    out.push(current >> 2);
    current = grid.cameFrom[current];
    guard += 1;
  }
  out.reverse();
  return out;
}

function simplify(points: WirePoint[]): WirePoint[] {
  const deduped: WirePoint[] = [];
  for (const point of points) {
    const last = deduped[deduped.length - 1];
    if (last && Math.abs(last.x - point.x) < EPS && Math.abs(last.y - point.y) < EPS) {
      continue;
    }
    deduped.push(point);
  }
  if (deduped.length <= 2) {
    return deduped;
  }
  const out: WirePoint[] = [deduped[0]];
  for (let k = 1; k < deduped.length - 1; k += 1) {
    const a = out[out.length - 1];
    const b = deduped[k];
    const c = deduped[k + 1];
    const collinear =
      (Math.abs(a.x - b.x) < EPS && Math.abs(b.x - c.x) < EPS) ||
      (Math.abs(a.y - b.y) < EPS && Math.abs(b.y - c.y) < EPS);
    if (!collinear) {
      out.push(b);
    }
  }
  out.push(deduped[deduped.length - 1]);
  return out;
}

/** Plain orthogonal path for when no clean route exists (a card fully boxed in). */
function fallbackPath(
  sourcePort: WirePoint,
  sourceStub: WirePoint,
  targetStub: WirePoint,
  targetPort: WirePoint,
): WirePoint[] {
  const midX = (sourceStub.x + targetStub.x) / 2;
  return simplify([
    sourcePort,
    sourceStub,
    { x: midX, y: sourceStub.y },
    { x: midX, y: targetStub.y },
    targetStub,
    targetPort,
  ]);
}

function edgeKey(
  edge: WireEdgeInput,
  sourceRect: WireRect,
  targetRect: WireRect,
): string {
  return [
    edge.sourceSide,
    edge.targetSide,
    edge.sourceOffset,
    edge.targetOffset,
    sourceRect.x,
    sourceRect.y,
    sourceRect.width,
    sourceRect.height,
    targetRect.x,
    targetRect.y,
    targetRect.width,
    targetRect.height,
  ].join(",");
}

type Segment = {
  edgeIndex: number;
  index: number;
  net: number;
  sink: number;
  vertical: boolean;
  coord: number;
  lo: number;
  hi: number;
  fixed: boolean;
};

function collectSegments(paths: WirePoint[][], owners: Owner[]): Segment[] {
  const segments: Segment[] = [];
  paths.forEach((points, edgeIndex) => {
    for (let k = 0; k + 1 < points.length; k += 1) {
      const a = points[k];
      const b = points[k + 1];
      if (Math.abs(a.x - b.x) < EPS && Math.abs(a.y - b.y) < EPS) {
        continue;
      }
      const vertical = Math.abs(a.x - b.x) < EPS;
      segments.push({
        edgeIndex,
        index: k,
        net: owners[edgeIndex].net,
        sink: owners[edgeIndex].sink,
        vertical,
        coord: vertical ? a.x : a.y,
        lo: vertical ? Math.min(a.y, b.y) : Math.min(a.x, b.x),
        hi: vertical ? Math.max(a.y, b.y) : Math.max(a.x, b.x),
        // The first and last segments leave and enter a port, so they cannot move sideways.
        fixed: k === 0 || k + 2 === points.length,
      });
    }
  });
  return segments;
}

function byLine(segments: Segment[]): Map<string, Segment[]> {
  const groups = new Map<string, Segment[]>();
  for (const segment of segments) {
    const key = `${segment.vertical ? "v" : "h"}:${Math.round(segment.coord * 2) / 2}`;
    const list = groups.get(key);
    if (list) {
      list.push(segment);
    } else {
      groups.set(key, [segment]);
    }
  }
  return groups;
}

/** Wires that run along the same line as an unrelated wire. */
function findOverlaps(paths: WirePoint[][], owners: Owner[]): number[] {
  const out = new Set<number>();
  for (const line of byLine(collectSegments(paths, owners)).values()) {
    line.sort((a, b) => a.lo - b.lo);
    for (let a = 0; a < line.length; a += 1) {
      const sa = line[a];
      for (let b = a + 1; b < line.length && line[b].lo < sa.hi - 1; b += 1) {
        const sb = line[b];
        if (sa.edgeIndex === sb.edgeIndex || sa.net === sb.net || sa.sink === sb.sink) {
          continue;
        }
        if (Math.min(sa.hi, sb.hi) - Math.max(sa.lo, sb.lo) > 1) {
          out.add(sa.edgeIndex);
          out.add(sb.edgeIndex);
        }
      }
    }
  }
  return [...out];
}

/**
 * Splits a long first or last leg into a fixed stub plus a movable remainder joined by a
 * zero-length jog, so two wires whose ports face each other can still take separate lanes.
 */
function withPortJogs(points: WirePoint[], stub: number): WirePoint[] {
  if (points.length < 2) {
    return points;
  }
  const out = [...points];
  const legLength = (a: WirePoint, b: WirePoint): number =>
    Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
  const towards = (a: WirePoint, b: WirePoint, distance: number): WirePoint => ({
    x: a.x + Math.sign(b.x - a.x) * distance,
    y: a.y + Math.sign(b.y - a.y) * distance,
  });
  // A leg of exactly stub length still gets a (zero-length) jog: otherwise a lane shift on the
  // next segment would move the stub's corner and bend the stub off its port.
  if (legLength(out[0], out[1]) > stub - EPS) {
    const jog = towards(out[0], out[1], stub);
    out.splice(1, 0, jog, { ...jog });
  }
  const last = out.length - 1;
  if (legLength(out[last - 1], out[last]) > stub - EPS) {
    const jog = towards(out[last], out[last - 1], stub);
    out.splice(last, 0, jog, { ...jog });
  }
  return out;
}

function assignLanes(
  paths: WirePoint[][],
  owners: Owner[],
  laneGap: number,
  maxOffset: number,
): WirePoint[][] {
  const groups = byLine(collectSegments(paths, owners));

  const offsets = new Map<Segment, number>();
  for (const group of groups.values()) {
    if (group.length < 2) {
      continue;
    }
    group.sort((a, b) => a.lo - b.lo);
    // Clusters of segments whose spans overlap.
    let cluster: Segment[] = [];
    let clusterHi = Number.NEGATIVE_INFINITY;
    const flush = (): void => {
      // Segments sharing an output or an input are one bundle and keep one lane.
      const bundle = cluster.map((_, index) => index);
      const find = (index: number): number => {
        let root = index;
        while (bundle[root] !== root) {
          root = bundle[root];
        }
        bundle[index] = root;
        return root;
      };
      for (let a = 0; a < cluster.length; a += 1) {
        for (let b = a + 1; b < cluster.length; b += 1) {
          const sa = cluster[a];
          const sb = cluster[b];
          const touch = Math.min(sa.hi, sb.hi) - Math.max(sa.lo, sb.lo) > EPS;
          if (touch && (sa.net === sb.net || sa.sink === sb.sink)) {
            bundle[find(a)] = find(b);
          }
        }
      }
      const order: number[] = [];
      const fixedBundles = new Set<number>();
      cluster.forEach((segment, index) => {
        const root = find(index);
        if (!order.includes(root)) {
          order.push(root);
        }
        if (segment.fixed) {
          fixedBundles.add(root);
        }
      });
      if (order.length < 2) {
        return;
      }
      const movable = order.filter((root) => !fixedBundles.has(root));
      const slots = fixedBundles.size > 0 ? movable.length + 1 : movable.length;
      const gap = Math.min(laneGap, slots > 1 ? (maxOffset * 2) / (slots - 1) : laneGap);
      const bundleOffset = new Map<number, number>();
      if (fixedBundles.size > 0) {
        // Fixed wires keep the line; the rest alternate either side of it.
        movable.forEach((root, index) => {
          const step = Math.floor(index / 2) + 1;
          const sign = index % 2 === 0 ? 1 : -1;
          bundleOffset.set(root, Math.max(-maxOffset, Math.min(maxOffset, sign * step * gap)));
        });
      } else {
        movable.forEach((root, index) => {
          bundleOffset.set(root, (index - (movable.length - 1) / 2) * gap);
        });
      }
      cluster.forEach((segment, index) => {
        const offset = bundleOffset.get(find(index));
        if (!segment.fixed && offset) {
          offsets.set(segment, offset);
        }
      });
    };
    for (const segment of group) {
      if (cluster.length > 0 && segment.lo >= clusterHi - EPS) {
        flush();
        cluster = [];
        clusterHi = Number.NEGATIVE_INFINITY;
      }
      cluster.push(segment);
      clusterHi = Math.max(clusterHi, segment.hi);
    }
    if (cluster.length > 0) {
      flush();
    }
  }

  const out = paths.map((points) => points.map((point) => ({ ...point })));
  for (const [segment, offset] of offsets) {
    const points = out[segment.edgeIndex];
    const a = points[segment.index];
    const b = points[segment.index + 1];
    if (segment.vertical) {
      a.x += offset;
      b.x += offset;
    } else {
      a.y += offset;
      b.y += offset;
    }
  }
  return out;
}

function findHops(
  paths: WirePoint[][],
  owners: Owner[],
  radius: number,
): WirePoint[][] {
  const hops: WirePoint[][] = paths.map(() => []);
  const segments = collectSegments(paths, owners);
  const vertical = segments
    .filter((segment) => segment.vertical)
    .sort((a, b) => a.coord - b.coord);
  for (const h of segments) {
    if (h.vertical) {
      continue;
    }
    const from = h.lo + radius + 1;
    const to = h.hi - radius - 1;
    for (let k = lowerSegment(vertical, from); k < vertical.length; k += 1) {
      const v = vertical[k];
      if (v.coord >= to) {
        break;
      }
      if (v.coord <= from || h.net === v.net || h.sink === v.sink) {
        continue;
      }
      if (h.coord > v.lo + EPS && h.coord < v.hi - EPS) {
        hops[h.edgeIndex].push({ x: v.coord, y: h.coord });
      }
    }
  }
  return hops;
}

function lowerSegment(sorted: Segment[], coord: number): number {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid].coord < coord) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

/** Point where two wires of one net stop sharing a path, or null if they part at the port. */
function divergencePoint(a: WirePoint[], b: WirePoint[]): WirePoint | null {
  let ia = 0;
  let ib = 0;
  let pa = { ...a[0] };
  let pb = { ...b[0] };
  let shared = 0;
  while (ia + 1 < a.length && ib + 1 < b.length) {
    const na = a[ia + 1];
    const nb = b[ib + 1];
    const da = { x: Math.sign(na.x - pa.x), y: Math.sign(na.y - pa.y) };
    const db = { x: Math.sign(nb.x - pb.x), y: Math.sign(nb.y - pb.y) };
    if (da.x !== db.x || da.y !== db.y) {
      break;
    }
    const la = Math.abs(na.x - pa.x) + Math.abs(na.y - pa.y);
    const lb = Math.abs(nb.x - pb.x) + Math.abs(nb.y - pb.y);
    const step = Math.min(la, lb);
    shared += step;
    pa = { x: pa.x + da.x * step, y: pa.y + da.y * step };
    pb = { ...pa };
    if (la - step < EPS) {
      ia += 1;
    }
    if (lb - step < EPS) {
      ib += 1;
    }
  }
  if (shared < EPS) {
    return null;
  }
  const endA = a[a.length - 1];
  const endB = b[b.length - 1];
  const atEnd =
    (Math.abs(pa.x - endA.x) < EPS && Math.abs(pa.y - endA.y) < EPS) ||
    (Math.abs(pa.x - endB.x) < EPS && Math.abs(pa.y - endB.y) < EPS);
  return atEnd ? null : pa;
}

function findJunctions(paths: WirePoint[][], owners: Owner[]): WirePoint[][] {
  const junctions: WirePoint[][] = paths.map(() => []);
  const seen = new Set<string>();
  const addFor = (key: (owner: Owner) => number, reverse: boolean): void => {
    const groups = new Map<number, number[]>();
    owners.forEach((owner, index) => {
      const list = groups.get(key(owner));
      if (list) {
        list.push(index);
      } else {
        groups.set(key(owner), [index]);
      }
    });
    for (const members of groups.values()) {
      for (let x = 0; x < members.length; x += 1) {
        for (let y = x + 1; y < members.length; y += 1) {
          const a = paths[members[x]];
          const b = paths[members[y]];
          const point = reverse
            ? divergencePoint([...a].reverse(), [...b].reverse())
            : divergencePoint(a, b);
          if (!point) {
            continue;
          }
          const id = `${Math.round(point.x)}:${Math.round(point.y)}`;
          if (seen.has(id)) {
            continue;
          }
          seen.add(id);
          junctions[members[x]].push(point);
        }
      }
    }
  };
  addFor((owner) => owner.net, false);
  addFor((owner) => owner.sink, true);
  return junctions;
}

const DEFAULT_OPTIONS: Required<WireRouterOptions> = {
  clearance: 20,
  laneGap: 6,
  bendPenalty: 40,
  crossingPenalty: 60,
  overlapPenalty: 400,
  clearancePenalty: 150,
  nearPenalty: 3,
  ringPenalty: 2,
  hopRadius: 4,
  cellBudget: 400_000,
};

const SEARCH_WINDOW_PAD = 200;
const ZONE_PREFIX = "\u0000zone:";

function intersects(a: WireRect, b: { x0: number; y0: number; x1: number; y1: number }): boolean {
  return a.x < b.x1 && a.x + a.width > b.x0 && a.y < b.y1 && a.y + a.height > b.y0;
}

const BUCKET = 512;

/** Rect indices by coarse grid bucket, so large graphs don't scan every card per wire. */
function bucketIndex(rects: WireRect[]): (box: { x0: number; y0: number; x1: number; y1: number }) => number[] {
  const buckets = new Map<number, number[]>();
  const keyOf = (bx: number, by: number): number => bx * 65_536 + by;
  rects.forEach((rect, index) => {
    for (let bx = Math.floor(rect.x / BUCKET); bx <= Math.floor((rect.x + rect.width) / BUCKET); bx += 1) {
      for (let by = Math.floor(rect.y / BUCKET); by <= Math.floor((rect.y + rect.height) / BUCKET); by += 1) {
        const key = keyOf(bx, by);
        const list = buckets.get(key);
        if (list) {
          list.push(index);
        } else {
          buckets.set(key, [index]);
        }
      }
    }
  });
  return (box) => {
    const found = new Set<number>();
    for (let bx = Math.floor(box.x0 / BUCKET); bx <= Math.floor(box.x1 / BUCKET); bx += 1) {
      for (let by = Math.floor(box.y0 / BUCKET); by <= Math.floor(box.y1 / BUCKET); by += 1) {
        for (const index of buckets.get(keyOf(bx, by)) ?? []) {
          found.add(index);
        }
      }
    }
    return [...found].sort((a, b) => a - b);
  };
}

function contains(rect: WireRect, point: WirePoint): boolean {
  return (
    point.x > rect.x &&
    point.x < rect.x + rect.width &&
    point.y > rect.y &&
    point.y < rect.y + rect.height
  );
}

/**
 * Routes every wire around the cards (`rects`) and through only the zones (group frames) it has an
 * end in. Pass the previous call's cache so only wires near a moved card are searched again.
 */
export function routeWires(
  rects: WireRect[],
  edges: WireEdgeInput[],
  options: WireRouterOptions = {},
  previous?: WireRouterCache,
  zones: WireRect[] = [],
): WireRouterResult {
  const costs = { ...DEFAULT_OPTIONS, ...options };
  const clearance = costs.clearance;
  const rectById = new Map(rects.map((rect) => [rect.id, rect] as const));
  const obstacles = rects.map((rect) => inflate(rect, clearance));
  const near = bucketIndex(obstacles);
  const tracked = new Map(rectById);
  for (const zone of zones) {
    tracked.set(ZONE_PREFIX + zone.id, zone);
  }

  const changed: WireRect[] = [];
  if (previous) {
    for (const [id, rect] of tracked) {
      const before = previous.rects.get(id);
      if (!sameRect(before, rect)) {
        changed.push(inflate(rect, clearance));
        if (before) {
          changed.push(inflate(before, clearance));
        }
      }
    }
    for (const [id, before] of previous.rects) {
      if (!tracked.has(id)) {
        changed.push(inflate(before, clearance));
      }
    }
  }

  const netIds = new Map<string, number>();
  const sinkIds = new Map<string, number>();
  type Prepared = {
    edge: WireEdgeInput;
    owner: Owner;
    exempt: WireRect[];
    zones: number[];
    key: string;
    sourcePort: WirePoint;
    targetPort: WirePoint;
    sourceStub: WirePoint;
    targetStub: WirePoint;
    cached: RawRoute | null;
  };
  const prepared: Prepared[] = [];
  const extraXs: number[] = [];
  const extraYs: number[] = [];
  for (const edge of edges) {
    const sourceRect = rectById.get(edge.source);
    const targetRect = rectById.get(edge.target);
    if (!(sourceRect && targetRect)) {
      continue;
    }
    const sink = edge.sink ?? `${edge.target}:${edge.targetSide}`;
    if (!netIds.has(edge.net)) {
      netIds.set(edge.net, netIds.size);
    }
    if (!sinkIds.has(sink)) {
      sinkIds.set(sink, sinkIds.size);
    }
    // Snapped to the grid's half-pixel steps so the first and last legs stay straight.
    const sourcePort = onHalfPixel(
      wirePortPoint(sourceRect, edge.sourceSide, edge.sourceOffset),
    );
    const targetPort = onHalfPixel(
      wirePortPoint(targetRect, edge.targetSide, edge.targetOffset),
    );
    const sourceStub = offsetPoint(sourcePort, SIDE_DIR[edge.sourceSide], clearance);
    const targetStub = offsetPoint(targetPort, SIDE_DIR[edge.targetSide], clearance);
    const key = edgeKey(edge, sourceRect, targetRect);
    const old = previous?.raw.get(edge.id);
    let cached: RawRoute | null = null;
    if (old && old.key === key && !old.blocked) {
      let touched = false;
      for (let k = 0; k + 1 < old.points.length && !touched; k += 1) {
        touched = changed.some((rect) =>
          segmentHitsRect(old.points[k], old.points[k + 1], rect),
        );
      }
      cached = touched ? null : old;
    }
    if (cached) {
      for (const point of cached.points) {
        extraXs.push(point.x);
        extraYs.push(point.y);
      }
    }
    extraXs.push(sourcePort.x, sourceStub.x, targetPort.x, targetStub.x);
    extraYs.push(sourcePort.y, sourceStub.y, targetPort.y, targetStub.y);
    const ends = [sourcePort, sourceStub, targetPort, targetStub];
    prepared.push({
      edge,
      owner: {
        net: netIds.get(edge.net) as number,
        sink: sinkIds.get(sink) as number,
      },
      exempt: near({
        x0: Math.min(...ends.map((p) => p.x)),
        y0: Math.min(...ends.map((p) => p.y)),
        x1: Math.max(...ends.map((p) => p.x)),
        y1: Math.max(...ends.map((p) => p.y)),
      })
        .map((k) => rects[k])
        .filter(
          (rect) =>
            rect.id !== edge.source &&
            rect.id !== edge.target &&
            ends.some((point) => contains(rect, point)),
        ),
      zones: zones.flatMap((zone, index) =>
        ends.some((point) => contains(zone, point)) ? [index] : [],
      ),
      key,
      sourcePort,
      targetPort,
      sourceStub,
      targetStub,
      cached,
    });
  }

  // Short wires first: they have the fewest options and should get the straight lines.
  const order = prepared
    .map((_, index) => index)
    .filter((index) => !prepared[index].cached)
    .sort((a, b) => {
      const pa = prepared[a];
      const pb = prepared[b];
      const la =
        Math.abs(pa.sourceStub.x - pa.targetStub.x) +
        Math.abs(pa.sourceStub.y - pa.targetStub.y);
      const lb =
        Math.abs(pb.sourceStub.x - pb.targetStub.x) +
        Math.abs(pb.sourceStub.y - pb.targetStub.y);
      return la - lb;
    });

  const raw: RawRoute[] = prepared.map(
    (item) => item.cached ?? { key: item.key, points: [], blocked: false },
  );

  const searchOn = (
    grid: Grid,
    item: Prepared,
    windows: SearchInput["window"][],
    itemZones: number[],
    searchCosts: Required<WireRouterOptions>,
  ): WirePoint[] | null => {
    const start =
      indexOf(grid.xs, item.sourceStub.x) * grid.ny + indexOf(grid.ys, item.sourceStub.y);
    const goal =
      indexOf(grid.xs, item.targetStub.x) * grid.ny + indexOf(grid.ys, item.targetStub.y);
    const startDir = SIDE_DIR[item.edge.sourceSide];
    // The last leg runs from the stub into the card, opposite to the side's outward direction.
    const goalDir = (SIDE_DIR[item.edge.targetSide] + 2) % 4;
    for (const window of windows) {
      const cells = searchPath(
        grid,
        {
          start,
          startDir,
          goal,
          goalDir,
          owner: item.owner,
          window,
          exempt: item.exempt,
          zones: itemZones,
        },
        searchCosts,
      );
      if (cells) {
        return simplify([
          item.sourcePort,
          ...cells.map((cell) => ({
            x: grid.xs[Math.floor(cell / grid.ny)],
            y: grid.ys[cell % grid.ny],
          })),
          item.targetPort,
        ]);
      }
    }
    return null;
  };

  const blockedRoute = (item: Prepared): RawRoute => ({
    key: item.key,
    points: fallbackPath(item.sourcePort, item.sourceStub, item.targetStub, item.targetPort),
    blocked: true,
  });

  const lineCount =
    sortedUnique([...extraXs, ...obstacles.flatMap((r) => [r.x, r.x + r.width])]).length *
    sortedUnique([...extraYs, ...obstacles.flatMap((r) => [r.y, r.y + r.height])]).length;

  if (lineCount * 2 > costs.cellBudget) {
    // Too big for one grid: each wire gets a grid of just the cards around it. Wires no longer see
    // each other while searching; lanes, hops and junctions below still apply to all of them.
    for (const index of order) {
      const item = prepared[index];
      let points: WirePoint[] | null = null;
      for (const pad of [SEARCH_WINDOW_PAD, SEARCH_WINDOW_PAD * 4]) {
        const box = {
          x0: Math.min(item.sourceStub.x, item.targetStub.x) - pad,
          x1: Math.max(item.sourceStub.x, item.targetStub.x) + pad,
          y0: Math.min(item.sourceStub.y, item.targetStub.y) - pad,
          y1: Math.max(item.sourceStub.y, item.targetStub.y) + pad,
        };
        const local: WireRect[] = [];
        const localObstacles: WireRect[] = [];
        for (const k of near(box)) {
          if (intersects(obstacles[k], box)) {
            local.push(rects[k]);
            localObstacles.push(obstacles[k]);
          }
        }
        const localZones: WireRect[] = [];
        const itemZones: number[] = [];
        zones.forEach((zone, k) => {
          if (intersects(zone, box)) {
            if (item.zones.includes(k)) {
              itemZones.push(localZones.length);
            }
            localZones.push(zone);
          }
        });
        const ends = [item.sourcePort, item.sourceStub, item.targetPort, item.targetStub];
        const grid = buildGrid(
          local,
          localObstacles,
          localZones,
          ends.map((p) => p.x),
          ends.map((p) => p.y),
          clearance * 2,
        );
        points = searchOn(
          grid,
          item,
          [{ i0: 0, i1: grid.nx - 1, j0: 0, j1: grid.ny - 1 }],
          itemZones,
          costs,
        );
        if (points) {
          break;
        }
      }
      raw[index] = points ? { key: item.key, points, blocked: false } : blockedRoute(item);
    }
  } else {
    const grid = buildGrid(rects, obstacles, zones, extraXs, extraYs, clearance * 2);
    for (const item of prepared) {
      const at = (point: WirePoint): number =>
        indexOf(grid.xs, point.x) * grid.ny + indexOf(grid.ys, point.y);
      claim(grid.reservedNet, at(item.sourceStub), item.owner.net);
      claim(grid.reservedSink, at(item.targetStub), item.owner.sink);
    }
    for (const item of prepared) {
      if (item.cached) {
        markPath(grid, item.cached.points, item.owner);
      }
    }

    const routeOne = (index: number, searchCosts: Required<WireRouterOptions>): void => {
      const item = prepared[index];
      const lowX = Math.min(item.sourceStub.x, item.targetStub.x) - SEARCH_WINDOW_PAD;
      const highX = Math.max(item.sourceStub.x, item.targetStub.x) + SEARCH_WINDOW_PAD;
      const lowY = Math.min(item.sourceStub.y, item.targetStub.y) - SEARCH_WINDOW_PAD;
      const highY = Math.max(item.sourceStub.y, item.targetStub.y) + SEARCH_WINDOW_PAD;
      const points = searchOn(
        grid,
        item,
        [
          {
            i0: Math.max(0, lowerIndex(grid.xs, lowX) - 1),
            i1: Math.min(grid.nx - 1, lowerIndex(grid.xs, highX)),
            j0: Math.max(0, lowerIndex(grid.ys, lowY) - 1),
            j1: Math.min(grid.ny - 1, lowerIndex(grid.ys, highY)),
          },
          { i0: 0, i1: grid.nx - 1, j0: 0, j1: grid.ny - 1 },
        ],
        item.zones,
        searchCosts,
      );
      if (!points) {
        raw[index] = blockedRoute(item);
        return;
      }
      markPath(grid, points, item.owner);
      raw[index] = { key: item.key, points, blocked: false };
    };

    for (const index of order) {
      routeOne(index, costs);
    }

    // Rip up and reroute: a wire that still overlaps an unrelated one is routed again with every
    // other wire in place and a steeper overlap cost.
    const rerouteCosts = { ...costs, overlapPenalty: costs.overlapPenalty * 3 };
    for (let pass = 0; pass < 4; pass += 1) {
      const conflicted = findOverlaps(
        raw.map((route) => route.points),
        prepared.map((item) => item.owner),
      );
      if (conflicted.length === 0) {
        break;
      }
      for (const index of conflicted) {
        grid.netH.fill(-1);
        grid.netV.fill(-1);
        grid.sinkH.fill(-1);
        grid.sinkV.fill(-1);
        prepared.forEach((item, other) => {
          if (other !== index && !raw[other].blocked) {
            markPath(grid, raw[other].points, item.owner);
          }
        });
        routeOne(index, rerouteCosts);
      }
    }
  }

  const owners = prepared.map((item) => item.owner);
  const laned = assignLanes(
    raw.map((route) => withPortJogs(route.points, clearance)),
    owners,
    costs.laneGap,
    clearance - 6,
  ).map(simplify);
  const hops = findHops(laned, owners, costs.hopRadius);
  const junctions = findJunctions(laned, owners);

  const routes = new Map<string, WireRoute>();
  const cache: WireRouterCache = { rects: tracked, raw: new Map() };
  prepared.forEach((item, index) => {
    routes.set(item.edge.id, {
      id: item.edge.id,
      points: laned[index],
      hops: hops[index],
      junctions: junctions[index],
      blocked: raw[index].blocked,
      ends: item.edge.ends ?? [item.edge.source, item.edge.target],
    });
    cache.raw.set(item.edge.id, raw[index]);
  });
  return { routes, cache, rerouted: order.length };
}

function lowerIndex(values: number[], value: number): number {
  let lo = 0;
  let hi = values.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (values[mid] < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}


/**
 * SVG path for a routed wire: every bend rounded with up to `corner` px (0 keeps square corners),
 * and a small arc wherever it hops another wire.
 */
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

/** Closest wire within `tolerance` of a point, with the y of the run when it is horizontal there. */
export function findWireNear(
  routes: Iterable<WireRoute>,
  point: WirePoint,
  tolerance: number,
): { id: string; y: number | null } | null {
  let best: { id: string; y: number | null; distance: number } | null = null;
  for (const route of routes) {
    const points = route.points;
    for (let k = 0; k + 1 < points.length; k += 1) {
      const a = points[k];
      const b = points[k + 1];
      const horizontal = Math.abs(a.y - b.y) < EPS;
      const along = horizontal
        ? point.x >= Math.min(a.x, b.x) && point.x <= Math.max(a.x, b.x)
        : point.y >= Math.min(a.y, b.y) && point.y <= Math.max(a.y, b.y);
      const distance = horizontal ? Math.abs(point.y - a.y) : Math.abs(point.x - a.x);
      if (along && distance <= tolerance && (!best || distance < best.distance)) {
        best = { id: route.id, y: horizontal ? a.y : null, distance };
      }
    }
  }
  return best ? { id: best.id, y: best.y } : null;
}
