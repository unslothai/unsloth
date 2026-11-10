// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Orthogonal A* wire router ("Bus" wires), ported from Recipe Studio. Group frames are zones only
// wires with an end inside may enter; graphs too big for one grid route each wire on a local grid.

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
  net: string;
  /** Wires into the same input handle may merge before it. Defaults to target + side. */
  sink?: string;
  /** Port position down the side from the card's top edge; the middle of the side if unset. */
  sourceOffset?: number;
  targetOffset?: number;
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
  /** Hard gap between a wire and any card it doesn't start or end on. */
  hardClearance?: number;
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
  blockedH: Uint8Array;
  blockedV: Uint8Array;
  softH: Uint8Array;
  softV: Uint8Array;
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
  value = 1,
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
          point[p] = value;
        }
        if (insideY && i < i1) {
          horizontal[p] = value;
        }
        if (insideX && j < j1) {
          vertical[p] = value;
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
        // Borders count as inside, so a wire that may not enter can't run along the frame either.
        const insideX = i >= i0 && i <= i1;
        const insideY = j >= j0 && j <= j1;
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
  const outer = margins.map((rect) => inflate(rect, OUTER_BAND));
  for (const rect of [...cards, ...margins, ...zones]) {
    rawXs.push(rect.x, rect.x + rect.width);
    rawYs.push(rect.y, rect.y + rect.height);
    minX = Math.min(minX, rect.x);
    minY = Math.min(minY, rect.y);
    maxX = Math.max(maxX, rect.x + rect.width);
    maxY = Math.max(maxY, rect.y + rect.height);
  }
  for (const rect of outer) {
    rawXs.push(rect.x, rect.x + rect.width);
    rawYs.push(rect.y, rect.y + rect.height);
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
  const buffers = gridBuffers(size);
  const blockedPoint = buffers.blockedPoint;
  const blockedH = buffers.blockedH;
  const blockedV = buffers.blockedV;
  const softH = buffers.softH;
  const softV = buffers.softV;
  markRects(cards, xs, ys, blockedPoint, blockedH, blockedV);
  // Two bands: close to a card costs most, the outer band a little, so a run takes the middle
  // of a gap between cards rather than either edge of it.
  markRects(outer, xs, ys, null, softH, softV, 1);
  markRects(margins, xs, ys, null, softH, softV, 2);
  const { zonePoint, zoneH, zoneV } = buffers;
  markZones(zones, xs, ys, zonePoint, zoneH, zoneV);

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
    netH: buffers.netH,
    netV: buffers.netV,
    sinkH: buffers.sinkH,
    sinkV: buffers.sinkV,
    reservedNet: buffers.reservedNet,
    reservedSink: buffers.reservedSink,
    inner: { x0: minX, x1: maxX, y0: minY, y1: maxY },
    stamp: buffers.stamp,
    closed: buffers.closed,
    gScore: buffers.gScore,
    cameFrom: buffers.cameFrom,
    search: buffers.search,
  };
}

type GridPool = {
  capacity: number;
  bytes: Uint8Array[];
  ints: Int32Array[];
  stamp: Int32Array;
  closed: Int32Array;
  gScore: Float64Array;
  cameFrom: Int32Array;
};

let pool: GridPool | null = null;
// Searches number on across grids, so stale stamps in a reused buffer never match a new search.
let searchCounter = 0;

/** Grids made and buffer sets allocated, for the perf tests. */
export const wireGridStats = { built: 0, allocated: 0 };

// Only one grid is alive at a time, so every pass reuses one set of buffers, grown when too small.
function gridBuffers(size: number) {
  wireGridStats.built += 1;
  if (!pool || pool.capacity < size || searchCounter > 0x3fffffff) {
    const capacity = Math.max(size, pool ? pool.capacity : 0);
    wireGridStats.allocated += 1;
    pool = {
      capacity,
      bytes: Array.from({ length: 5 }, () => new Uint8Array(capacity)),
      ints: Array.from({ length: 9 }, () => new Int32Array(capacity)),
      stamp: new Int32Array(capacity * 4),
      closed: new Int32Array(capacity * 4),
      gScore: new Float64Array(capacity * 4),
      cameFrom: new Int32Array(capacity * 4),
    };
    searchCounter = 0;
  }
  const bytes = pool.bytes.map((values) => values.subarray(0, size).fill(0));
  const ints = pool.ints.map((values) => values.subarray(0, size).fill(-1));
  return {
    blockedPoint: bytes[0],
    blockedH: bytes[1],
    blockedV: bytes[2],
    softH: bytes[3],
    softV: bytes[4],
    zonePoint: ints[0],
    zoneH: ints[1],
    zoneV: ints[2],
    netH: ints[3],
    netV: ints[4],
    sinkH: ints[5],
    sinkV: ints[6],
    reservedNet: ints[7],
    reservedSink: ints[8],
    stamp: pool.stamp.subarray(0, size * 4),
    closed: pool.closed.subarray(0, size * 4),
    gScore: pool.gScore.subarray(0, size * 4),
    cameFrom: pool.cameFrom.subarray(0, size * 4),
    search: searchCounter,
  };
}

function claim(values: Int32Array, p: number, id: number): void {
  const current = values[p];
  values[p] = current === -1 || current === id ? id : -2;
}

function forEachUnit(
  grid: Grid,
  points: WirePoint[],
  visit: (vertical: boolean, p: number) => void,
): void {
  const { ny } = grid;
  for (let k = 0; k + 1 < points.length; k += 1) {
    const ai = indexOf(grid.xs, points[k].x);
    const aj = indexOf(grid.ys, points[k].y);
    const bi = indexOf(grid.xs, points[k + 1].x);
    const bj = indexOf(grid.ys, points[k + 1].y);
    if (ai < 0 || aj < 0 || bi < 0 || bj < 0 || (ai !== bi && aj !== bj)) {
      continue;
    }
    if (aj === bj) {
      for (let i = Math.min(ai, bi); i < Math.max(ai, bi); i += 1) {
        visit(false, i * ny + aj);
      }
    } else {
      for (let j = Math.min(aj, bj); j < Math.max(aj, bj); j += 1) {
        visit(true, ai * ny + j);
      }
    }
  }
}

function markPath(grid: Grid, points: WirePoint[], owner: Owner): void {
  forEachUnit(grid, points, (vertical, p) => {
    claim(vertical ? grid.netV : grid.netH, p, owner.net);
    claim(vertical ? grid.sinkV : grid.sinkH, p, owner.sink);
  });
}

// Owners per segment, so one wire's marks can be removed without re-marking the rest.
class Claims {
  private owners = new Map<number, Owner[]>();
  private grid: Grid;

  constructor(grid: Grid) {
    this.grid = grid;
  }

  add(points: WirePoint[], owner: Owner): void {
    forEachUnit(this.grid, points, (vertical, p) => {
      const key = p * 2 + (vertical ? 1 : 0);
      const list = this.owners.get(key);
      if (list) {
        list.push(owner);
      } else {
        this.owners.set(key, [owner]);
      }
    });
  }

  remove(points: WirePoint[], owner: Owner): void {
    const { grid } = this;
    forEachUnit(grid, points, (vertical, p) => {
      const list = this.owners.get(p * 2 + (vertical ? 1 : 0));
      const at = list ? list.indexOf(owner) : -1;
      if (!list || at < 0) {
        return;
      }
      list.splice(at, 1);
      const net = vertical ? grid.netV : grid.netH;
      const sink = vertical ? grid.sinkV : grid.sinkH;
      net[p] = -1;
      sink[p] = -1;
      for (const other of list) {
        claim(net, p, other.net);
        claim(sink, p, other.sink);
      }
    });
  }
}

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
  searchCounter = Math.max(searchCounter, grid.search) + 1;
  grid.search = searchCounter;
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
      // A stub that starts inside a closed zone may leave it, never enter another.
      const here = q === goal ? grid.zonePoint[q] : grid.zonePoint[p];
      if (walled(grid.zonePoint[q]) && grid.zonePoint[q] !== here) {
        continue;
      }
      const midX = (xs[ni] + xs[i]) / 2;
      const midY = (ys[nj] + ys[j]) / 2;
      let segNet: number;
      let segSink: number;
      let soft: number;
      if (nd === 0 || nd === 2) {
        const left = nd === 0 ? p : q;
        if ((grid.blockedH[left] && !inExempt(exempt, midX, midY)) || (walled(grid.zoneH[left]) && grid.zoneH[left] !== here)) {
          continue;
        }
        segNet = grid.netH[left];
        segSink = grid.sinkH[left];
        soft = grid.softH[left];
      } else {
        const upper = nd === 1 ? p : q;
        if ((grid.blockedV[upper] && !inExempt(exempt, midX, midY)) || (walled(grid.zoneV[upper]) && grid.zoneV[upper] !== here)) {
          continue;
        }
        segNet = grid.netV[upper];
        segSink = grid.sinkV[upper];
        soft = grid.softV[upper];
      }
      const length = Math.abs(xs[ni] - xs[i]) + Math.abs(ys[nj] - ys[j]);
      let cost =
        soft === 2 ? length * 2 + costs.clearancePenalty : soft === 1 ? length * 1.3 : length;
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
        cost += costs.overlapPenalty + length * 0.5;
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
        // Never double back into the stub from the card side; arriving along the side costs a bend.
        if (nd === (goalDir + 2) % 4) {
          continue;
        }
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

// A zero-length jog after the stub lets wires with facing ports take separate lanes.
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
  // Even a stub-length leg gets a jog, or a lane shift would bend the stub off its port.
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
  roomFor: (segment: Segment) => number,
): WirePoint[][] {
  const groups = byLine(collectSegments(paths, owners));

  const offsets = new Map<Segment, number>();
  // Runs on one line that end this close together read as one wire, so they cluster like overlaps.
  const endGap = laneGap * 3;
  const near = (sa: Segment, sb: Segment): boolean =>
    Math.min(sa.hi, sb.hi) - Math.max(sa.lo, sb.lo) > -endGap;
  for (const group of groups.values()) {
    if (group.length < 2) {
      continue;
    }
    group.sort((a, b) => a.lo - b.lo);
    let cluster: Segment[] = [];
    let clusterHi = Number.NEGATIVE_INFINITY;
    const flush = (): void => {
      const bundle = cluster.map((_, index) => index);
      const find = (index: number): number => {
        let root = index;
        while (bundle[root] !== root) {
          root = bundle[root];
        }
        bundle[index] = root;
        return root;
      };
      const nets = cluster.map((segment) => new Set([segment.net]));
      const sinks = cluster.map((segment) => new Set([segment.sink]));
      // Never chain a fan-out and a fan-in: two unrelated wires would read as one.
      const join = (a: number, b: number): void => {
        const ra = find(a);
        const rb = find(b);
        if (ra === rb) {
          return;
        }
        const n = new Set([...nets[ra], ...nets[rb]]);
        const k = new Set([...sinks[ra], ...sinks[rb]]);
        if (n.size > 1 && k.size > 1) {
          return;
        }
        bundle[ra] = rb;
        nets[rb] = n;
        sinks[rb] = k;
      };
      for (let a = 0; a < cluster.length; a += 1) {
        for (let b = a + 1; b < cluster.length; b += 1) {
          const sa = cluster[a];
          const sb = cluster[b];
          if (near(sa, sb) && sa.net === sb.net) {
            join(a, b);
          }
        }
      }
      for (let a = 0; a < cluster.length; a += 1) {
        for (let b = a + 1; b < cluster.length; b += 1) {
          const sa = cluster[a];
          const sb = cluster[b];
          if (near(sa, sb) && sa.sink === sb.sink) {
            join(a, b);
          }
        }
      }
      const order: number[] = [];
      const fixedBundles = new Set<number>();
      cluster.forEach((_, index) => {
        const root = find(index);
        if (!order.includes(root)) {
          order.push(root);
        }
      });
      // A port leg pins its bundle only where another bundle runs beside it.
      cluster.forEach((segment, index) => {
        if (
          segment.fixed &&
          cluster.some((other, k) => find(k) !== find(index) && near(segment, other))
        ) {
          fixedBundles.add(find(index));
        }
      });
      if (order.length < 2) {
        return;
      }
      const movable = order.filter((root) => !fixedBundles.has(root));
      // Fixed port legs never move, so only the runs that do bound the spread.
      const maxOffset = Math.max(
        0,
        Math.min(...cluster.filter((segment) => !segment.fixed).map(roomFor)),
      );
      const slots = fixedBundles.size > 0 ? movable.length + 1 : movable.length;
      const gap = Math.min(laneGap, slots > 1 ? (maxOffset * 2) / (slots - 1) : laneGap);
      const bundleOffset = new Map<number, number>();
      if (fixedBundles.size > 0) {
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
      if (cluster.length > 0 && segment.lo >= clusterHi + endGap) {
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
    const key = segment.vertical ? "x" : "y";
    // Never shift a run so far that the run before or after it turns back on itself.
    let shift = offset;
    for (const [end, step] of [
      [a, -1],
      [b, 1],
    ] as const) {
      let at = step < 0 ? segment.index - 1 : segment.index + 2;
      let length = 0;
      while (points[at] && Math.abs(length) < EPS && Math.abs(at - segment.index) < 5) {
        length = end[key] - points[at][key];
        at += step;
      }
      if (length > EPS) {
        shift = Math.max(shift, Math.min(0, 4 - length));
      } else if (length < -EPS) {
        shift = Math.min(shift, Math.max(0, -4 - length));
      }
    }
    a[key] += shift;
    b[key] += shift;
  }
  return out;
}

function findHops(
  paths: WirePoint[][],
  owners: Owner[],
  radius: number,
  junctions: WirePoint[] = [],
): WirePoint[][] {
  const hops: WirePoint[][] = paths.map(() => []);
  const segments = collectSegments(paths, owners);
  const vertical = segments
    .filter((segment) => segment.vertical)
    .sort((a, b) => a.coord - b.coord);
  // A wire of h's own net or sink on v's line at the crossing means they join there: no hop.
  const joins = (h: Segment, v: Segment): boolean =>
    vertical.some(
      (o) =>
        Math.abs(o.coord - v.coord) < EPS &&
        (o.net === h.net || o.sink === h.sink) &&
        h.coord >= o.lo - EPS &&
        h.coord <= o.hi + EPS,
    );
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
      // Only a run that clears the arc on both sides is a crossing; a short lane jog is not.
      if (
        h.coord > v.lo + radius &&
        h.coord < v.hi - radius &&
        !joins(h, v) &&
        // A fan-out or fan-in dot inside the arc would read as sitting on the hop.
        !junctions.some(
          (j) => Math.abs(j.y - h.coord) < EPS && Math.abs(j.x - v.coord) <= radius * 2,
        )
      ) {
        hops[h.edgeIndex].push({ x: v.coord, y: h.coord });
      }
    }
  }
  // Wires bundled on one line cross at one point: one arc, and none crowding the last.
  return hops.map((list) => {
    const kept: WirePoint[] = [];
    for (const hop of list.sort((a, b) => a.y - b.y || a.x - b.x)) {
      const last = kept[kept.length - 1];
      if (!last || Math.abs(last.y - hop.y) > EPS || hop.x - last.x > radius * 2 + 1) {
        kept.push(hop);
      }
    }
    return kept;
  });
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

type FanIn = { port: WirePoint; side: WireSide; sink: number };

// Fan-in wires share the last leg from the stub so they never hop or lane-split at the port.
function mergeFanIn(
  paths: WirePoint[][],
  ends: FanIn[],
  owners: Owner[],
  distance: number,
  blocked: (index: number, a: WirePoint, b: WirePoint) => boolean,
): WirePoint[][] {
  // A moved run must not land on an unrelated wire's lane, including one moved just before it.
  const result = [...paths];
  let lanes = collectSegments(result, owners);
  const onLane = (index: number, a: WirePoint, b: WirePoint): boolean => {
    const vertical = Math.abs(a.x - b.x) < EPS;
    const coord = vertical ? a.x : a.y;
    const lo = vertical ? Math.min(a.y, b.y) : Math.min(a.x, b.x);
    const hi = vertical ? Math.max(a.y, b.y) : Math.max(a.x, b.x);
    const { net, sink } = owners[index];
    return lanes.some(
      (lane) =>
        lane.edgeIndex !== index &&
        lane.net !== net &&
        lane.sink !== sink &&
        lane.vertical === vertical &&
        Math.abs(lane.coord - coord) < 1 &&
        Math.min(lane.hi, hi) - Math.max(lane.lo, lo) > 1,
    );
  };
  const count = new Map<number, number>();
  for (const end of ends) {
    count.set(end.sink, (count.get(end.sink) ?? 0) + 1);
  }
  const rewrite = (path: WirePoint[], index: number): WirePoint[] => {
    const { port, side, sink } = ends[index];
    if ((count.get(sink) ?? 0) < 2 || path.length < 3) {
      return path;
    }
    const out = DIRS[SIDE_DIR[side]];
    const d = { x: -out.x, y: -out.y };
    const along = (p: WirePoint): number => p.x * d.x + p.y * d.y;
    const across = (p: WirePoint): number => (d.x !== 0 ? p.y : p.x);
    const make = (a: number, c: number): WirePoint =>
      d.x !== 0 ? { x: a * d.x, y: c } : { x: c, y: a * d.y };
    const points = simplify(path);
    const stub = along(port) - distance;
    let k = points.length - 1;
    while (k >= 0 && Math.abs(across(points[k]) - across(port)) < EPS) {
      k -= 1;
    }
    if (k < 1) {
      return points;
    }
    const q = points[k];
    const prev = points[k - 1];
    const landed = points[k + 1];
    if (
      Math.abs(along(landed) - stub) < EPS ||
      Math.abs(across(prev) - across(q)) > EPS ||
      (stub - along(prev)) * (along(q) - along(prev)) <= EPS
    ) {
      return points;
    }
    const corner = make(stub, across(q));
    const merge = make(stub, across(port));
    if (
      blocked(index, prev, corner) ||
      blocked(index, corner, merge) ||
      onLane(index, prev, corner) ||
      onLane(index, corner, merge)
    ) {
      return points;
    }
    return simplify([...points.slice(0, k), corner, merge, port]);
  };
  result.forEach((path, index) => {
    const next = rewrite(path, index);
    if (next !== path) {
      result[index] = next;
      lanes = collectSegments(result, owners);
    }
  });
  return result;
}

type Walls = { cards: WireRect[]; zones: WireRect[] };

function channelAround(
  walls: Walls,
  vertical: boolean,
  coord: number,
  lo: number,
  hi: number,
): { low: number; high: number } | null {
  let low = Number.NEGATIVE_INFINITY;
  let high = Number.POSITIVE_INFINITY;
  const across = (rect: WireRect): [number, number] =>
    vertical ? [rect.x, rect.x + rect.width] : [rect.y, rect.y + rect.height];
  const along = (rect: WireRect): [number, number] =>
    vertical ? [rect.y, rect.y + rect.height] : [rect.x, rect.x + rect.width];
  for (const rect of walls.cards) {
    const [a0, a1] = along(rect);
    if (a1 <= lo + EPS || a0 >= hi - EPS) {
      continue;
    }
    const [c0, c1] = across(rect);
    if (coord > c0 + EPS && coord < c1 - EPS) {
      // Already crossing a card it is allowed through; leave it be.
      return null;
    }
    if (c1 <= coord + EPS) {
      low = Math.max(low, c1);
    } else {
      high = Math.min(high, c0);
    }
  }
  for (const zone of walls.zones) {
    const [a0, a1] = along(zone);
    if (a1 <= lo + EPS || a0 >= hi - EPS) {
      continue;
    }
    for (const edge of across(zone)) {
      if (Math.abs(edge - coord) < EPS) {
        continue;
      }
      if (edge < coord) {
        low = Math.max(low, edge);
      } else {
        high = Math.min(high, edge);
      }
    }
  }
  return { low, high };
}

function snapNearRuns(
  paths: WirePoint[][],
  owners: Owner[],
  tolerance: number,
  fits: (segment: Segment, coord: number) => boolean,
): WirePoint[][] {
  const out = paths.map((points) => points.map((point) => ({ ...point })));
  for (const vertical of [true, false]) {
    const runs = collectSegments(out, owners)
      .filter((segment) => !segment.fixed && segment.vertical === vertical)
      .sort((a, b) => a.coord - b.coord);
    const key = vertical ? "x" : "y";
    for (let a = 0; a < runs.length; a += 1) {
      for (let b = a + 1; b < runs.length && runs[b].coord - runs[a].coord < tolerance; b += 1) {
        const ra = runs[a];
        const rb = runs[b];
        if (
          Math.abs(rb.coord - ra.coord) < EPS ||
          Math.min(ra.hi, rb.hi) - Math.max(ra.lo, rb.lo) <= EPS ||
          !fits(rb, ra.coord)
        ) {
          continue;
        }
        const points = out[rb.edgeIndex];
        points[rb.index][key] = ra.coord;
        points[rb.index + 1][key] = ra.coord;
        rb.coord = ra.coord;
      }
    }
  }
  return out;
}

// Parallel runs in one channel pack at lane pitch around its middle, in order.
function runSpan(runs: Segment[]): { lo: number; hi: number } {
  return {
    lo: Math.min(...runs.map((run) => run.lo)),
    hi: Math.max(...runs.map((run) => run.hi)),
  };
}

function gatherChannels(
  paths: WirePoint[][],
  owners: Owner[],
  walls: Walls,
  pitch: number,
  hard: number,
  stub: number,
): WirePoint[][] {
  const out = paths.map((points) => points.map((point) => ({ ...point })));
  const blocks = walls.cards.map((rect) => inflate(rect, hard - 1));
  for (const vertical of [true, false]) {
    const key = vertical ? "x" : "y";
    const groups = new Map<string, Segment[]>();
    const all = collectSegments(out, owners);
    for (const run of all) {
      if (run.fixed || run.vertical !== vertical) {
        continue;
      }
      const gap = channelAround(walls, vertical, run.coord, run.lo, run.hi);
      if (!gap || !Number.isFinite(gap.low) || !Number.isFinite(gap.high)) {
        continue;
      }
      const id = `${Math.round(gap.low)}:${Math.round(gap.high)}`;
      groups.set(id, [...(groups.get(id) ?? []), run]);
    }
    for (const [id, runs] of groups) {
      const [low, high] = id.split(":").map(Number);
      runs.sort((a, b) => a.lo - b.lo);
      // Split into bundles whose spans overlap, so runs at opposite ends of a long channel stay apart.
      const bundles: Segment[][] = [];
      let reach = Number.NEGATIVE_INFINITY;
      for (const run of runs) {
        if (run.lo < reach - EPS) {
          bundles[bundles.length - 1].push(run);
        } else {
          bundles.push([run]);
        }
        reach = Math.max(reach, run.hi);
      }
      for (const bundle of bundles) {
        const coords = sortedUnique(bundle.map((run) => run.coord));
        if (coords.length < 2 || high - low - 2 * hard < (coords.length - 1) * pitch) {
          continue;
        }
        const first = (low + high) / 2 - ((coords.length - 1) * pitch) / 2;
        const target = new Map(coords.map((coord, k) => [coord, Math.round((first + k * pitch) * 2) / 2]));
        const moves = bundle.map((run) => ({ run, to: target.get(run.coord) as number }));
        const crowds = all.some(
          (other) =>
            other.vertical === vertical &&
            !bundle.includes(other) &&
            Math.min(other.hi, runSpan(bundle).hi) - Math.max(other.lo, runSpan(bundle).lo) > EPS &&
            moves.some(({ to }) => Math.abs(other.coord - to) < pitch - EPS),
        );
        const keeps = !crowds && moves.every(({ run, to }) => {
          const points = out[run.edgeIndex];
          const before = points[run.index];
          const after = points[run.index + 1];
          const prev = points[run.index - 1];
          const next = points[run.index + 2];
          const legOk = (from: WirePoint, at: WirePoint, k: number): boolean => {
            const was = at[key] - from[key];
            const now = to - from[key];
            if (Math.abs(was) < EPS || Math.sign(was) !== Math.sign(now)) {
              return false;
            }
            const isPortLeg = k === 0 || k + 2 === points.length;
            if (isPortLeg && Math.abs(now) < stub - EPS) {
              return false;
            }
            const moved = { ...at, [key]: to };
            // The leg into the moved run grows; it must not reach into a card it used to clear.
            return blocks.every(
              (rect) =>
                (isPortLeg && contains(rect, from)) || !segmentHitsRect(from, moved, rect),
            );
          };
          return legOk(prev, before, run.index - 1) && legOk(next, after, run.index + 1);
        });
        if (!keeps) {
          continue;
        }
        for (const { run, to } of moves) {
          out[run.edgeIndex][run.index][key] = to;
          out[run.edgeIndex][run.index + 1][key] = to;
        }
      }
    }
  }
  return out;
}

const OUTER_BAND = 20;

const DEFAULT_OPTIONS: Required<WireRouterOptions> = {
  clearance: 24,
  hardClearance: 18,
  laneGap: 9,
  bendPenalty: 40,
  crossingPenalty: 60,
  overlapPenalty: 40,
  // Per grid step in a card's clearance band; small, since a tight channel splits into many steps.
  clearancePenalty: 12,
  nearPenalty: 0,
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

// Bucketed so large graphs don't scan every card per wire.
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

function clearanceSide(rect: WireRect, band: WireRect, point: WirePoint): WireRect {
  const right = rect.x + rect.width;
  const bottom = rect.y + rect.height;
  if (point.x >= right) {
    return { ...band, x: right, width: band.x + band.width - right };
  }
  if (point.x <= rect.x) {
    return { ...band, width: rect.x - band.x };
  }
  if (point.y <= rect.y) {
    return { ...band, height: rect.y - band.y };
  }
  return { ...band, y: bottom, height: band.y + band.height - bottom };
}

function contains(rect: WireRect, point: WirePoint): boolean {
  return (
    point.x > rect.x &&
    point.x < rect.x + rect.width &&
    point.y > rect.y &&
    point.y < rect.y + rect.height
  );
}

/** Pass the previous call's cache so only wires near a moved card are searched again. */
export function routeWires(
  rects: WireRect[],
  edges: WireEdgeInput[],
  options: WireRouterOptions = {},
  previous?: WireRouterCache,
  zones: WireRect[] = [],
): WireRouterResult {
  const costs = { ...DEFAULT_OPTIONS, ...options };
  const clearance = costs.clearance;
  const hard = Math.min(costs.hardClearance, clearance - 2);
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
      }).flatMap((k) => {
        const rect = rects[k];
        if (rect.id === edge.source || rect.id === edge.target) {
          return [];
        }
        const band = inflate(rect, hard);
        const inside = ends.filter((point) => contains(band, point));
        if (inside.length === 0) {
          return [];
        }
        // A card over the port frees the whole card; one merely near it frees only its band.
        return inside.some((point) => contains(rect, point))
          ? [band]
          : inside.map((point) => clearanceSide(rect, band, point));
      }),
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

  const localRoute = (item: Prepared, by: number): WirePoint[] | null => {
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
        local.map((rect) => inflate(rect, by)),
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
    return points;
  };

  if (lineCount * 2 > costs.cellBudget) {
    // Too big for one grid: local grids; lanes, hops and junctions below still apply to all.
    for (const index of order) {
      const item = prepared[index];
      const points = localRoute(item, hard);
      raw[index] = points ? { key: item.key, points, blocked: false } : blockedRoute(item);
    }
  } else {
    // Cards grow by the hard clearance: no wire may run closer than that to one.
    const grid = buildGrid(
      rects.map((rect) => inflate(rect, hard)),
      obstacles,
      zones,
      extraXs,
      extraYs,
      clearance * 2,
    );
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

    const routeOne = (
      index: number,
      searchCosts: Required<WireRouterOptions>,
      claims?: Claims,
    ): void => {
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
      claims?.add(points, item.owner);
      raw[index] = { key: item.key, points, blocked: false };
    };

    for (const index of order) {
      routeOne(index, costs);
    }

    // Rip up and reroute overlapping wires with a steeper overlap cost.
    const rerouteCosts = { ...costs, overlapPenalty: costs.overlapPenalty * 3 };
    let claims: Claims | null = null;
    for (let pass = 0; pass < 4; pass += 1) {
      const conflicted = findOverlaps(
        raw.map((route) => route.points),
        prepared.map((item) => item.owner),
      );
      if (conflicted.length === 0) {
        break;
      }
      if (!claims) {
        claims = new Claims(grid);
        for (let other = 0; other < raw.length; other += 1) {
          if (!raw[other].blocked) {
            claims.add(raw[other].points, prepared[other].owner);
          }
        }
      }
      for (const index of conflicted) {
        if (!raw[index].blocked) {
          claims.remove(raw[index].points, prepared[index].owner);
        }
        routeOne(index, rerouteCosts, claims);
      }
    }
  }

  // Cards closer than twice the clearance leave no legal gap: squeeze through instead.
  for (const index of order) {
    if (raw[index].blocked) {
      const points = localRoute(prepared[index], 0);
      if (points) {
        raw[index] = { key: prepared[index].key, points, blocked: false };
      }
    }
  }

  const walls: Walls = { cards: rects, zones };
  const owners = prepared.map((item) => item.owner);
  const fanIn: FanIn[] = prepared.map((item) => ({
    port: item.targetPort,
    side: item.edge.targetSide,
    sink: item.owner.sink,
  }));
  const fanInBlocked = (index: number, a: WirePoint, b: WirePoint): boolean => {
    const { edge, zones: own } = prepared[index];
    return (
      rects.some(
        (rect) =>
          rect.id !== edge.target &&
          segmentHitsRect(a, b, rect.id === edge.source ? rect : inflate(rect, hard)),
      ) || zones.some((zone, z) => !own.includes(z) && segmentHitsRect(a, b, zone))
    );
  };
  const merged = mergeFanIn(
    raw.map((route) => route.points),
    fanIn,
    owners,
    clearance,
    fanInBlocked,
  );
  const jogged = merged.map((points) => withPortJogs(points, clearance));
  const roomFor = (segment: Segment): number => {
    if (segment.fixed) {
      return 0;
    }
    const gap = channelAround(walls, segment.vertical, segment.coord, segment.lo, segment.hi);
    if (!gap) {
      return costs.laneGap;
    }
    const room = Math.min(segment.coord - gap.low, gap.high - segment.coord) - hard;
    return Math.min(room, costs.laneGap * 4);
  };
  // Twice: spreading one set of runs can stretch the runs between them into a new overlap.
  let laned = snapNearRuns(jogged, owners, costs.laneGap, (segment, coord) => {
    const gap = channelAround(walls, segment.vertical, coord, segment.lo, segment.hi);
    return !gap || Math.min(coord - gap.low, gap.high - coord) >= hard - EPS;
  });
  for (let pass = 0; pass < 2; pass += 1) {
    laned = assignLanes(laned, owners, costs.laneGap, roomFor).map(simplify);
  }
  laned = gatherChannels(laned, owners, walls, costs.laneGap, hard, clearance).map(simplify);
  laned = mergeFanIn(laned, fanIn, owners, clearance, fanInBlocked);
  const junctions = findJunctions(laned, owners);
  const hops = findHops(laned, owners, costs.hopRadius, junctions.flat());

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


