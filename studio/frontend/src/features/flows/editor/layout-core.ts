// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import dagre from "@dagrejs/dagre";

export interface LayoutRequest {
  nodes: { id: string; width: number; height: number }[];
  edges: { source: string; target: string }[];
  direction: "LR" | "TB";
  nodesep?: number;
  ranksep?: number;
  /** Fold a layout wider (LR) or taller (TB) than this ratio into bands, so a long chain wraps. */
  wrapAspect?: number;
}

export type LayoutResult = Record<string, [number, number]>;

export function computeLayout(request: LayoutRequest): LayoutResult {
  const graph = new dagre.graphlib.Graph();
  graph.setGraph({
    rankdir: request.direction,
    nodesep: request.nodesep ?? 40,
    ranksep: request.ranksep ?? 90,
    marginx: 20,
    marginy: 20,
  });
  graph.setDefaultEdgeLabel(() => ({}));
  for (const node of request.nodes)
    graph.setNode(node.id, { width: node.width, height: node.height });
  for (const edge of request.edges) {
    if (graph.hasNode(edge.source) && graph.hasNode(edge.target))
      graph.setEdge(edge.source, edge.target);
  }
  dagre.layout(graph);
  const result: LayoutResult = {};
  for (const node of request.nodes) {
    const laid = graph.node(node.id) as { x: number; y: number } | undefined;
    if (laid)
      result[node.id] = [
        Math.round(laid.x - node.width / 2),
        Math.round(laid.y - node.height / 2),
      ];
  }
  return request.wrapAspect ? wrapBands(request, result, request.wrapAspect) : result;
}

type Box = { id: string; x: number; y: number; w: number; h: number };

/** Cuts the rank columns (rows for TB) into bands of about equal length and stacks the bands. */
export function wrapBands(
  request: LayoutRequest,
  result: LayoutResult,
  aspect: number,
): LayoutResult {
  const lr = request.direction !== "TB";
  // Measured along the flow (main) and across it (cross).
  const boxes: Box[] = request.nodes
    .filter((n) => result[n.id])
    .map((n) => {
      const [x, y] = result[n.id];
      return lr
        ? { id: n.id, x, y, w: n.width, h: n.height }
        : { id: n.id, x: y, y: x, w: n.height, h: n.width };
    });
  if (boxes.length < 4) return result;
  const left = Math.min(...boxes.map((b) => b.x));
  const right = Math.max(...boxes.map((b) => b.x + b.w));
  const top = Math.min(...boxes.map((b) => b.y));
  const bottom = Math.max(...boxes.map((b) => b.y + b.h));
  const bands = Math.round(
    Math.sqrt((right - left) / (Math.max(bottom - top, 1) * aspect)),
  );
  if (bands < 2) return result;
  // A rank shares its center along the flow.
  const ranks = new Map<number, Box[]>();
  for (const b of boxes) {
    const key = Math.round(b.x + b.w / 2);
    ranks.set(key, [...(ranks.get(key) ?? []), b]);
  }
  const columns = [...ranks.entries()].sort((a, b) => a[0] - b[0]).map((e) => e[1]);
  const target = (right - left) / bands;
  const groups: Box[][][] = [[]];
  for (const column of columns) {
    const band = groups[groups.length - 1];
    const start = band.length ? Math.min(...band.flat().map((b) => b.x)) : 0;
    const end = Math.max(...column.map((b) => b.x + b.w));
    if (band.length && end - start > target && groups.length < bands) groups.push([column]);
    else band.push(column);
  }
  const gap = request.ranksep ?? 90;
  const out: LayoutResult = {};
  let cursor = top;
  for (const band of groups) {
    const flat = band.flat();
    const bandLeft = Math.min(...flat.map((b) => b.x));
    const bandTop = Math.min(...flat.map((b) => b.y));
    const bandBottom = Math.max(...flat.map((b) => b.y + b.h));
    for (const b of flat) {
      const main = Math.round(b.x - bandLeft + left);
      const cross = Math.round(b.y - bandTop + cursor);
      out[b.id] = lr ? [main, cross] : [cross, main];
    }
    cursor += bandBottom - bandTop + gap;
  }
  return out;
}
