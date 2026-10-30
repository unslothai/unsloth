// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Tidy up, minus the DOM: what goes to the layout worker, and where everything lands after.
// A group is laid out as one block, so its cards keep their spots inside it and the frame keeps its
// size. Notes and empty frames ride along with the card nearest them. Reroutes are not laid out:
// they are put back on the new wire, between the ends they join.

import type { DocGraph, TidyPatch } from "../stores/doc-store";
import {
  type RerouteDir,
  rerouteDownstream,
  rerouteUpstream,
  wireInto,
} from "../stores/reroute-graph";
import type { LayoutRequest, LayoutResult } from "./layout-core";

type Size = { width: number; height: number };
type Rect = [number, number, number, number];

export const GROUP_BLOCK = "group:";
const DEFAULT_CARD: Size = { width: 260, height: 140 };
const REROUTE_BOX: Size = { width: 28, height: 16 };

/** Bus wires need lanes between columns and rows; curves do not. */
export const TIDY_SPACING = {
  bus: { nodesep: 60, ranksep: 140 },
  curved: { nodesep: 40, ranksep: 100 },
} as const;

export interface TidyContext {
  graph: DocGraph;
  /** A card's size once fitted. */
  size: (id: string) => Size;
  /** A port's height from the top of its card. */
  portY: (id: string, port: string, side: "in" | "out") => number | undefined;
  wireStyle: "bus" | "curved";
}

function members(graph: DocGraph): Map<string, string[]> {
  const out = new Map<string, string[]>();
  for (const id of graph.nodeOrder) {
    const group = graph.nodes[id].group;
    if (!group || !graph.groups[group]) continue;
    out.set(group, [...(out.get(group) ?? []), id]);
  }
  return out;
}

function inside(rect: Rect, outer: Rect): boolean {
  return (
    rect[0] >= outer[0] &&
    rect[1] >= outer[1] &&
    rect[0] + rect[2] <= outer[0] + outer[2] &&
    rect[1] + rect[3] <= outer[1] + outer[3]
  );
}

function gap(a: Rect, b: Rect): number {
  const dx = Math.max(0, a[0] - (b[0] + b[2]), b[0] - (a[0] + a[2]));
  const dy = Math.max(0, a[1] - (b[1] + b[3]), b[1] - (a[1] + a[3]));
  return Math.hypot(dx, dy);
}

export function tidyRequest(
  ctx: TidyContext,
  direction: "LR" | "TB" = "LR",
): LayoutRequest {
  const { graph } = ctx;
  const grouped = members(graph);
  const owner = new Map<string, string>();
  for (const [group, ids] of grouped)
    for (const id of ids) owner.set(id, GROUP_BLOCK + group);
  const blockOf = (id: string) => owner.get(id) ?? id;
  const nodes = graph.nodeOrder
    .filter((id) => !owner.has(id))
    .map((id) => ({ id, ...ctx.size(id) }));
  for (const group of grouped.keys()) {
    const rect = graph.groups[group].rect;
    nodes.push({ id: GROUP_BLOCK + group, width: rect[2], height: rect[3] });
  }
  // The plain edges already run end to end through any reroutes.
  const seen = new Set<string>();
  const edges = graph.edgeOrder.flatMap((id) => {
    const source = blockOf(graph.edges[id].source);
    const target = blockOf(graph.edges[id].target);
    const key = `${source}->${target}`;
    if (source === target || seen.has(key)) return [];
    seen.add(key);
    return [{ source, target }];
  });
  return { nodes, edges, direction, ...TIDY_SPACING[ctx.wireStyle] };
}

export function tidyPatch(
  ctx: TidyContext,
  result: LayoutResult,
  sizes: Record<string, [number, number] | null> = {},
): TidyPatch {
  const { graph } = ctx;
  const patch: TidyPatch = { nodes: {}, groups: {}, notes: {}, reroutes: {} };
  const grouped = members(graph);
  const shift = new Map<string, [number, number]>();
  for (const id of graph.nodeOrder) {
    const at = result[id];
    if (at) shift.set(id, [at[0] - graph.nodes[id].pos[0], at[1] - graph.nodes[id].pos[1]]);
  }
  for (const [group, ids] of grouped) {
    const at = result[GROUP_BLOCK + group];
    if (!at) continue;
    const rect = graph.groups[group].rect;
    const d: [number, number] = [at[0] - rect[0], at[1] - rect[1]];
    patch.groups[group] = [at[0], at[1], rect[2], rect[3]];
    shift.set(GROUP_BLOCK + group, d);
    for (const id of ids) shift.set(id, d);
  }
  for (const [id, d] of shift) {
    if (id.startsWith(GROUP_BLOCK)) continue;
    const [x, y] = graph.nodes[id].pos;
    patch.nodes[id] = {
      pos: [Math.round(x + d[0]), Math.round(y + d[1])],
      ...(id in sizes ? { size: sizes[id] } : {}),
    };
  }

  // Notes and empty frames: inside a moved frame they move with it, else with the nearest card.
  const cardRect = (id: string): Rect => {
    const [x, y] = graph.nodes[id].pos;
    const size = ctx.size(id);
    return [x, y, size.width, size.height];
  };
  const carry = (rect: Rect): [number, number] | null => {
    for (const group of grouped.keys())
      if (inside(rect, graph.groups[group].rect))
        return shift.get(GROUP_BLOCK + group) ?? null;
    let best: string | null = null;
    let bestGap = Number.POSITIVE_INFINITY;
    for (const id of graph.nodeOrder) {
      const d = gap(rect, cardRect(id));
      if (d < bestGap) [best, bestGap] = [id, d];
    }
    return best ? (shift.get(best) ?? null) : null;
  };
  for (const id of graph.groupOrder) {
    if (grouped.has(id)) continue;
    const rect = graph.groups[id].rect;
    const d = carry(rect);
    if (d) patch.groups[id] = [rect[0] + d[0], rect[1] + d[1], rect[2], rect[3]];
  }
  for (const id of graph.noteOrder) {
    const rect = graph.notes[id].rect;
    const d = carry(rect);
    if (d) patch.notes[id] = [rect[0] + d[0], rect[1] + d[1], rect[2], rect[3]];
  }

  Object.assign(patch.reroutes, placeReroutes(ctx, patch));
  return patch;
}

function rerouteSizeOf(dir: RerouteDir | undefined): Size {
  return dir === "down" || dir === "up"
    ? { width: REROUTE_BOX.height, height: REROUTE_BOX.width }
    : REROUTE_BOX;
}

/** Steps from a reroute back to its real source, and on to the nearest real input. */
function depth(graph: DocGraph, id: string): [number, number] {
  let back = 0;
  let cursor = id;
  const seen = new Set<string>();
  while (graph.reroutes[cursor] && !seen.has(cursor)) {
    seen.add(cursor);
    const into = wireInto(graph, cursor);
    back++;
    if (!into) break;
    cursor = into.source;
  }
  let ahead = 0;
  let frontier = [id];
  const visited = new Set<string>();
  while (frontier.length && ahead < 64) {
    ahead++;
    const next: string[] = [];
    for (const at of frontier) {
      if (visited.has(at)) continue;
      visited.add(at);
      for (const wid of graph.wireOrder) {
        const wire = graph.wires[wid];
        if (wire.source !== at) continue;
        if (!graph.reroutes[wire.target]) return [back, ahead];
        next.push(wire.target);
      }
    }
    frontier = next;
  }
  return [back, ahead];
}

function placeReroutes(
  ctx: TidyContext,
  patch: TidyPatch,
): Record<string, [number, number]> {
  const { graph } = ctx;
  const posOf = (id: string) => patch.nodes[id]?.pos ?? graph.nodes[id].pos;
  const out: Record<string, [number, number]> = {};
  for (const id of graph.rerouteOrder) {
    const source = rerouteUpstream(graph, id);
    const targets = rerouteDownstream(graph, id).filter((t) => graph.nodes[t.node]);
    if (!source || !graph.nodes[source.node] || !targets.length) continue;
    const [sx0, sy0] = posOf(source.node);
    const sx = sx0 + ctx.size(source.node).width;
    const sy = sy0 + (ctx.portY(source.node, source.port, "out") ?? 40);
    const ends = targets.map((t) => {
      const [x, y] = posOf(t.node);
      return [x, y + (ctx.portY(t.node, t.port, "in") ?? 40)] as const;
    });
    const tx = Math.min(...ends.map((e) => e[0]));
    const ty = ends.length === 1 ? ends[0][1] : sy;
    const [back, ahead] = depth(graph, id);
    const t = back / (back + ahead);
    const reroute = graph.reroutes[id];
    const upright = reroute.dir === "down" || reroute.dir === "up";
    const cx = sx + (tx - sx) * t;
    // A bus wire runs level out of its port and turns after the reroute; a curve goes straight across.
    const cy = upright
      ? (sy + ty) / 2
      : ctx.wireStyle === "bus" || ends.length > 1
        ? sy
        : sy + (ty - sy) * t;
    const size = rerouteSizeOf(reroute.dir);
    out[id] = [Math.round(cx - size.width / 2), Math.round(cy - size.height / 2)];
  }
  return out;
}

/** Fits a card to its output: wide and tall enough to read, never past the cap. */
export const CARD_FIT_MAX = { width: 420, height: 460 } as const;

export function fittedCardSize(
  current: Size,
  hidden: number,
  slack: number,
  minWidth: number,
): [number, number] | null {
  if (hidden > 2) {
    const width =
      hidden > 60
        ? Math.min(CARD_FIT_MAX.width, Math.max(current.width, 340))
        : current.width;
    // Wider text wraps into fewer lines.
    const textW = (w: number) => Math.max(1, w - 48);
    const extra = hidden * (textW(current.width) / textW(width));
    return [
      Math.round(Math.max(width, minWidth)),
      Math.round(
        Math.max(current.height, Math.min(CARD_FIT_MAX.height, current.height + extra + 6)),
      ),
    ];
  }
  if (slack > 24)
    return [Math.round(current.width), Math.round(current.height - slack + 8)];
  return null;
}
