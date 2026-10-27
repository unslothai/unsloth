// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type DocGraph, newId, useDocStore } from "../stores/doc-store";
import type { FlowGroup, FlowNode } from "../types";
import {
  COLUMN_TYPE,
  GENERATE_TYPE,
  columnName,
  columnOf,
  groupColumns,
  isRecipeGroup,
  referenceEdgeDiff,
  starterRecipe,
  withReference,
  withoutReference,
} from "./recipe-columns";

const COLUMN_SIZE: [number, number] = [320, 180];

/** Why two nodes cannot be wired as recipe columns, or null when the rule does not apply / allows it. */
export function recipeConnectionError(
  graph: DocGraph,
  sourceId: string,
  targetId: string,
): string | null {
  const source = graph.nodes[sourceId];
  const target = graph.nodes[targetId];
  const columns = [source?.type, target?.type].filter((t) => t === COLUMN_TYPE);
  if (columns.length === 0) return null;
  if (columns.length === 1 || source?.group !== target?.group)
    return "A recipe column only wires to columns of its own recipe";
  return null;
}

/** Upsert columns and the group in one undo step, then redraw the group's reference wires. */
export function commitRecipe(
  groupId: string,
  nodes: FlowNode[],
  group: FlowGroup | null,
  coalesceKey?: string,
): void {
  const doc = useDocStore.getState();
  const after: DocGraph = {
    ...doc.graph,
    nodes: {
      ...doc.graph.nodes,
      ...Object.fromEntries(nodes.map((n) => [n.id, n])),
    },
    nodeOrder: [
      ...doc.graph.nodeOrder,
      ...nodes.filter((n) => !doc.graph.nodes[n.id]).map((n) => n.id),
    ],
  };
  const diff = referenceEdgeDiff(after, groupColumns(after, groupId));
  if (!nodes.length && !group && !diff.add.length && !diff.remove.length)
    return;
  doc.patchGraph(
    {
      nodes,
      groups: group ? [group] : [],
      addEdges: diff.add,
      removeEdges: diff.remove,
    },
    coalesceKey,
  );
}

/** Wiring column A into column B writes `{{ A }}` into B, so the wire is the reference. */
export function connectColumns(
  sourceId: string,
  targetId: string,
): string | null {
  const graph = useDocStore.getState().graph;
  const refused = recipeConnectionError(graph, sourceId, targetId);
  if (refused) return refused;
  const source = graph.nodes[sourceId];
  const target = graph.nodes[targetId];
  const next = withReference(columnOf(target), columnName(source));
  if (!next)
    return `${columnName(target)} picks what it reads in its editor; double-click it`;
  commitRecipe(
    target.group as string,
    [{ ...target, params: { ...target.params, column: next } }],
    null,
  );
  return null;
}

/** Removes the references behind deleted column wires; returns the edge ids it handled. */
export function disconnectColumns(edgeIds: string[]): string[] {
  const graph = useDocStore.getState().graph;
  const changed = new Map<string, FlowNode>();
  const handled: string[] = [];
  let groupId: string | null = null;
  for (const id of edgeIds) {
    const edge = graph.edges[id];
    const source = edge && graph.nodes[edge.source];
    const target =
      edge && (changed.get(edge.target) ?? graph.nodes[edge.target]);
    if (source?.type !== COLUMN_TYPE || target?.type !== COLUMN_TYPE) continue;
    handled.push(id);
    groupId = target.group ?? null;
    const column = withoutReference(columnOf(target), columnName(source));
    changed.set(target.id, { ...target, params: { ...target.params, column } });
  }
  // A wire whose reference survives (a sampler's parent) is drawn again by the sync.
  if (groupId) commitRecipe(groupId, [...changed.values()], null);
  return handled;
}

/** Deletes nodes, taking a deleted column's `{{ reference }}` out of the columns that read it. */
export function removeNodes(nodeIds: string[]): void {
  const graph = useDocStore.getState().graph;
  const drop = new Set(nodeIds);
  disconnectColumns(
    graph.edgeOrder.filter((id) => {
      const edge = graph.edges[id];
      return drop.has(edge.source) && !drop.has(edge.target);
    }),
  );
  useDocStore.getState().removeElements({ nodes: nodeIds });
}

export function insideRecipeGroup(
  graph: DocGraph,
  pos: [number, number],
): string | null {
  for (const id of graph.groupOrder) {
    const group = graph.groups[id];
    const [x, y, w, h] = group.rect;
    if (
      isRecipeGroup(group) &&
      pos[0] >= x &&
      pos[0] <= x + w &&
      pos[1] >= y &&
      pos[1] <= y + h
    )
      return id;
  }
  return null;
}

/** A new Data Recipe group at `pos`: a starter sampler and LLM column wired into a Recipe Dataset step. */
export function insertStarterRecipe(
  pos: [number, number],
  generateParams: Record<string, unknown>,
): FlowNode {
  const groupId = newId("recipe");
  const { config, columns } = starterRecipe();
  const nodes: FlowNode[] = columns.map((column, i) => ({
    id: `${groupId}_col_${i + 1}`,
    type: COLUMN_TYPE,
    v: 1,
    params: { column },
    pos: [pos[0] + i * COLUMN_SIZE[0], pos[1]],
    label: String(column.name),
    group: groupId,
  }));
  const generate: FlowNode = {
    id: `${groupId}_generate`,
    type: GENERATE_TYPE,
    v: 1,
    params: generateParams,
    pos: [pos[0] + columns.length * COLUMN_SIZE[0], pos[1]],
    group: groupId,
  };
  const group: FlowGroup = {
    id: groupId,
    label: "Data Recipe",
    color: "#3b82f6",
    rect: [
      pos[0] - 40,
      pos[1] - 60,
      (columns.length + 1) * COLUMN_SIZE[0] + 40,
      COLUMN_SIZE[1] + 60,
    ],
    kind: "recipe",
    config,
    run: { rows: 5 },
  };
  useDocStore
    .getState()
    .patchGraph({ groups: [group], nodes: [...nodes, generate] });
  commitRecipe(groupId, [], null);
  return generate;
}

/** Where the next column of a group goes: right of its columns, grown into the group frame. */
export function nextColumnSlot(
  graph: DocGraph,
  groupId: string,
): { pos: [number, number]; group: FlowGroup } {
  const group = graph.groups[groupId];
  const columns = groupColumns(graph, groupId);
  const [x, y, w, h] = group.rect;
  const bottom = columns.reduce(
    (max, n) => Math.max(max, n.pos[1]),
    y + 60 - COLUMN_SIZE[1],
  );
  const pos: [number, number] = [x + 40, bottom + COLUMN_SIZE[1]];
  const height = Math.max(h, pos[1] - y + COLUMN_SIZE[1]);
  return { pos, group: { ...group, rect: [x, y, w, height] } };
}
