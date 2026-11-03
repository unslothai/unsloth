// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "sonner";
import { type DocGraph, newId, useDocStore } from "../stores/doc-store";
import type { FlowGroup, FlowNode } from "../types";
import { CARD_SIZE, setupBoxes, setupLayout } from "./recipe-setup";
import {
  COLUMN_TYPE,
  GENERATE_TYPE,
  columnName,
  columnOf,
  groupColumns,
  isRecipeGroup,
  referenceEdgeDiff,
  starterRecipe,
  templateStillReads,
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

/** Upsert columns and the group in one undo step, then redraw the groups' reference wires. */
export function commitRecipe(
  groupId: string | string[],
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
  const diffs = [groupId].flat().map((id) =>
    referenceEdgeDiff(after, groupColumns(after, id)),
  );
  const diff = {
    add: diffs.flatMap((d) => d.add),
    remove: diffs.flatMap((d) => d.remove),
  };
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

type Disconnect = {
  changed: Map<string, FlowNode>;
  handled: string[];
  groupIds: Set<string>;
  /** "target reads source" for a reference only a `{% ... %}` tag holds, which is not rewritten. */
  blocked: string[];
};

function planDisconnect(graph: DocGraph, edgeIds: string[]): Disconnect {
  const plan: Disconnect = {
    changed: new Map(),
    handled: [],
    groupIds: new Set(),
    blocked: [],
  };
  for (const id of edgeIds) {
    const edge = graph.edges[id];
    const source = edge && graph.nodes[edge.source];
    const target =
      edge && (plan.changed.get(edge.target) ?? graph.nodes[edge.target]);
    if (source?.type !== COLUMN_TYPE || target?.type !== COLUMN_TYPE) continue;
    const ref = columnName(source);
    const column = withoutReference(columnOf(target), ref);
    if (templateStillReads(column, ref)) {
      plan.blocked.push(`${columnName(target)} reads ${ref}`);
      continue;
    }
    plan.handled.push(id);
    if (target.group) plan.groupIds.add(target.group);
    plan.changed.set(target.id, {
      ...target,
      params: { ...target.params, column },
    });
  }
  return plan;
}

function refuse(blocked: string[], what: string): void {
  toast.error(`Cannot ${what}: ${blocked.join(", ")} in a {% %} tag`, {
    description: "Edit that column's prompt to drop the tag, then try again.",
  });
}

function commitDisconnect(plan: Disconnect, coalesceKey?: string): void {
  // A wire whose reference survives (a sampler's parent) is drawn again by the sync.
  if (plan.groupIds.size)
    commitRecipe(
      [...plan.groupIds],
      [...plan.changed.values()],
      null,
      coalesceKey,
    );
}

/** Removes the references behind deleted column wires; returns the edge ids it handled. A reference a
 * `{% ... %}` tag holds is not rewritten: that wire stays, with a note on how to remove it. */
export function disconnectColumns(
  edgeIds: string[],
  coalesceKey?: string,
): string[] {
  const plan = planDisconnect(useDocStore.getState().graph, edgeIds);
  if (plan.blocked.length) refuse(plan.blocked, "remove the wire");
  commitDisconnect(plan, coalesceKey);
  return plan.handled;
}

/** The wires from deleted nodes into kept ones. */
function wiresOut(graph: DocGraph, nodeIds: string[]): string[] {
  const drop = new Set(nodeIds);
  return graph.edgeOrder.filter((id) => {
    const edge = graph.edges[id];
    return drop.has(edge.source) && !drop.has(edge.target);
  });
}

/** Deletes nodes, taking a deleted column's `{{ reference }}` out of the columns that read it. A column a
 * `{% ... %}` tag still reads is not deleted, so no template is left naming a missing column. */
export function removeNodes(nodeIds: string[]): void {
  const graph = useDocStore.getState().graph;
  const plan = planDisconnect(graph, wiresOut(graph, nodeIds));
  if (plan.blocked.length) return refuse(plan.blocked, "delete it");
  // One key for both commits, so a single Undo brings the column and its references back together.
  const key = `remove:${nodeIds.join(",")}:${Date.now()}`;
  commitDisconnect(plan, key);
  useDocStore.getState().removeElements({ nodes: nodeIds }, key);
}

/** The canvas's delete: removes a deleted column's references in the same undo step as the elements. */
export function removeElementsWithReferences(ids: {
  nodes: string[];
  edges: string[];
  groups: string[];
  notes: string[];
}): void {
  const graph = useDocStore.getState().graph;
  // A reader deleted too needs no rewrite: only the wires into columns that stay.
  const deleted = new Set(ids.nodes);
  const plan = planDisconnect(
    graph,
    [
      ...ids.edges,
      ...wiresOut(graph, ids.nodes).filter((id) => !ids.edges.includes(id)),
    ].filter((id) => !deleted.has(graph.edges[id]?.target)),
  );
  if (plan.blocked.length) return refuse(plan.blocked, "delete it");
  const key = `delete:${[...ids.nodes, ...ids.edges].join(",")}:${Date.now()}`;
  commitDisconnect(plan, key);
  const references = new Set(plan.handled);
  useDocStore.getState().removeElements(
    { ...ids, edges: ids.edges.filter((id) => !references.has(id)) },
    key,
  );
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

type Box = [number, number, number, number];

const overlaps = (a: Box, b: Box) =>
  a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3];

/** `pos` if a `size` box there covers nothing on the canvas, else the same x below everything. */
export function freeSpot(
  graph: DocGraph,
  pos: [number, number],
  size: [number, number],
  margin: [number, number] = [40, 60],
): [number, number] {
  const setup = setupLayout(graph, null, new Map());
  const boxes: Box[] = [
    ...graph.nodeOrder.map((id): Box => {
      const node = graph.nodes[id];
      return [node.pos[0], node.pos[1], COLUMN_SIZE[0] - 40, COLUMN_SIZE[1] - 40];
    }),
    // A recipe frame as drawn: grown to hold its setup cards.
    ...graph.groupOrder.map((id): Box => {
      const [x, y, w, h] = graph.groups[id].rect;
      return [x, y, w, Math.max(h, setup.heights[id] ?? 0)];
    }),
    ...graph.noteOrder.map((id) => graph.notes[id].rect as Box),
    // Setup cards are canvas cards but not graph nodes, wherever they were dragged.
    ...setup.elements.map(({ x, y }): Box => [x, y, CARD_SIZE.width, CARD_SIZE.height]),
  ];
  const want: Box = [pos[0] - margin[0], pos[1] - margin[1], size[0], size[1]];
  if (!boxes.some((box) => overlaps(want, box))) return pos;
  const bottom = Math.max(...boxes.map((box) => box[1] + box[3]));
  return [pos[0], Math.round(bottom + margin[1] + 80)];
}

/** A starter recipe group at `pos`: a topic sampler and an LLM column wired into a Recipe Dataset step. */
export function starterRecipeGraph(
  pos: [number, number],
  generateParams: Record<string, unknown>,
): { group: FlowGroup; nodes: FlowNode[]; generate: FlowNode } {
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
  return { group, nodes: [...nodes, generate], generate };
}

/** A starter recipe group at `at`, or the nearest free space below it. */
export function insertStarterRecipe(
  at: [number, number],
  generateParams: Record<string, unknown>,
): FlowNode {
  const graph = useDocStore.getState().graph;
  const columns = starterRecipe().columns.length;
  const pos = freeSpot(graph, at, [
    (columns + 1) * COLUMN_SIZE[0] + 40,
    COLUMN_SIZE[1] + 60,
  ]);
  const { group, nodes, generate } = starterRecipeGraph(pos, generateParams);
  useDocStore.getState().patchGraph({ groups: [group], nodes });
  commitRecipe(group.id, [], null);
  return generate;
}

/** Where the next column of a group goes: the first free cell of the frame's column grid, else a new row
 *  below its columns, the frame grown to hold it. The Recipe Dataset step's lane stays clear for its cards. */
export function nextColumnSlot(
  graph: DocGraph,
  groupId: string,
): { pos: [number, number]; group: FlowGroup } {
  const group = graph.groups[groupId];
  const members = graph.nodeOrder
    .map((id) => graph.nodes[id])
    .filter((n) => n.group === groupId);
  const step = members.find((n) => n.type === GENERATE_TYPE);
  const [x, y, w, h] = group.rect;
  const [cw, ch] = COLUMN_SIZE;
  const cards = setupBoxes(graph);
  const taken = (cx: number, cy: number) =>
    (step && Math.abs(cx - step.pos[0]) < cw && cy >= step.pos[1] - ch / 2) ||
    members.some(
      (n) => Math.abs(n.pos[0] - cx) < cw - 20 && Math.abs(n.pos[1] - cy) < ch - 20,
    ) ||
    // Clear of the seed, model and tool cards too, wherever they were dragged.
    cards.some(
      (b) =>
        cx < b.x + b.width && b.x < cx + cw - 40 && cy < b.y + b.height && b.y < cy + ch - 40,
    );
  for (let cy = y + 60; cy + ch - 40 <= y + h; cy += ch)
    for (let cx = x + 40; cx + cw - 40 <= x + w; cx += cw)
      if (!taken(cx, cy)) return { pos: [cx, cy], group };
  const bottom = members
    .filter((n) => n.type === COLUMN_TYPE)
    .reduce((max, n) => Math.max(max, n.pos[1]), y + 60 - ch);
  // A new row below the columns, past anything already there (a card dragged into it).
  let pos: [number, number] = [x + 40, bottom + ch];
  for (let i = 0; i < 200 && taken(pos[0], pos[1]); i++) {
    const free = (() => {
      for (let cx = x + 40; cx + cw - 40 <= x + w; cx += cw)
        if (!taken(cx, pos[1])) return cx;
      return null;
    })();
    if (free !== null) {
      pos = [free, pos[1]];
      break;
    }
    pos = [x + 40, pos[1] + ch];
  }
  const height = Math.max(h, pos[1] - y + ch);
  return { pos, group: { ...group, rect: [x, y, w, height] } };
}
