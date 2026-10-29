// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Edge, Node } from "@xyflow/react";
import { type DocGraph, GROUP_KEY, NOTE_KEY } from "../stores/doc-store";
import type { FlowEdge, FlowGroup, FlowNode, FlowNote } from "../types";
import { edgeHandles } from "./connection-rules";

export type FlowNodeData = { nodeId: string };
export type FlowEdgeData = { kind: "data" | "flow"; portType: string };

export type RfNode = Node<
  FlowNodeData | { groupId: string } | { noteId: string }
>;
export type RfEdge = Edge<FlowEdgeData>;

export const GROUP_PREFIX = GROUP_KEY;
export const NOTE_PREFIX = NOTE_KEY;

type Positions = Record<string, { x: number; y: number }>;
/** A size set live by a resize handle, before it is committed on release. */
export type Boxes = Record<string, { width: number; height: number }>;

/**
 * How many frames enclose each frame. Frames sit under the cards, nested ones above the frames
 * around them, so an outer frame drawn later never covers an inner frame's header.
 */
export function groupDepths(graph: DocGraph): Map<string, number> {
  const depths = new Map<string, number>();
  for (const id of graph.groupOrder) {
    const [x, y, w, h] = graph.groups[id].rect;
    let depth = 0;
    for (const other of graph.groupOrder) {
      if (other === id) continue;
      const [ox, oy, ow, oh] = graph.groups[other].rect;
      const inside = ox <= x && oy <= y && ox + ow >= x + w && oy + oh >= y + h;
      const same = ox === x && oy === y && ow === w && oh === h;
      // Identical frames stack in document order.
      if (
        inside &&
        (!same ||
          graph.groupOrder.indexOf(other) < graph.groupOrder.indexOf(id))
      )
        depth += 1;
    }
    depths.set(id, depth);
  }
  return depths;
}

const GROUP_Z = -100;

export class RfDeriver {
  private nodeCache = new Map<
    string,
    {
      source: FlowNode;
      selected: boolean;
      x: number;
      y: number;
      size: { width: number; height: number } | undefined;
      box: unknown;
      rf: RfNode;
    }
  >();
  private dataCache = new Map<string, FlowNodeData>();
  private groupCache = new Map<
    string,
    {
      source: FlowGroup;
      selected: boolean;
      x: number;
      y: number;
      size: unknown;
      box: unknown;
      rf: RfNode;
    }
  >();
  private noteCache = new Map<
    string,
    {
      source: FlowNote;
      selected: boolean;
      x: number;
      y: number;
      size: unknown;
      box: unknown;
      rf: RfNode;
    }
  >();
  private edgeCache = new Map<
    string,
    { source: FlowEdge; selected: boolean; portType: string; rf: RfEdge }
  >();
  private lastNodes: RfNode[] = [];
  private lastEdges: RfEdge[] = [];

  nodes(
    graph: DocGraph,
    selected: ReadonlySet<string>,
    drag: Positions | null,
    sizes?: ReadonlyMap<string, { width: number; height: number }>,
    boxes?: Boxes | null,
  ): RfNode[] {
    const next: RfNode[] = [];
    let changed = false;
    const seen = new Set<string>();
    const depths = graph.groupOrder.length > 1 ? groupDepths(graph) : undefined;
    for (const groupId of graph.groupOrder) {
      const group = graph.groups[groupId];
      const rfId = GROUP_PREFIX + groupId;
      seen.add(rfId);
      const pos = drag?.[rfId] ?? { x: group.rect[0], y: group.rect[1] };
      const isSelected = selected.has(rfId);
      const size = sizes?.get(rfId);
      const box = boxes?.[rfId];
      const zIndex = GROUP_Z + (depths?.get(groupId) ?? 0);
      const cached = this.groupCache.get(groupId);
      if (
        cached &&
        cached.rf.zIndex === zIndex &&
        cached.source === group &&
        cached.selected === isSelected &&
        cached.x === pos.x &&
        cached.y === pos.y &&
        cached.size === size &&
        cached.box === box
      ) {
        next.push(cached.rf);
        continue;
      }
      changed = true;
      const rf: RfNode = {
        id: rfId,
        type: "flowGroup",
        position: pos,
        data: { groupId },
        selected: isSelected,
        style: box ?? { width: group.rect[2], height: group.rect[3] },
        zIndex,
        dragHandle: ".flow-group-drag",
        ...(size ? { measured: size } : {}),
      };
      this.groupCache.set(groupId, {
        source: group,
        selected: isSelected,
        x: pos.x,
        y: pos.y,
        size,
        box,
        rf,
      });
      next.push(rf);
    }
    for (const noteId of graph.noteOrder) {
      const note = graph.notes[noteId];
      const rfId = NOTE_PREFIX + noteId;
      seen.add(rfId);
      const pos = drag?.[rfId] ?? { x: note.rect[0], y: note.rect[1] };
      const isSelected = selected.has(rfId);
      const size = sizes?.get(rfId);
      const box = boxes?.[rfId];
      const cached = this.noteCache.get(noteId);
      if (
        cached &&
        cached.source === note &&
        cached.selected === isSelected &&
        cached.x === pos.x &&
        cached.y === pos.y &&
        cached.size === size &&
        cached.box === box
      ) {
        next.push(cached.rf);
        continue;
      }
      changed = true;
      const rf: RfNode = {
        id: rfId,
        type: "flowNote",
        position: pos,
        data: { noteId },
        selected: isSelected,
        style: box ?? { width: note.rect[2], height: note.rect[3] },
        ...(size ? { measured: size } : {}),
      };
      this.noteCache.set(noteId, {
        source: note,
        selected: isSelected,
        x: pos.x,
        y: pos.y,
        size,
        box,
        rf,
      });
      next.push(rf);
    }
    for (const id of graph.nodeOrder) {
      const node = graph.nodes[id];
      seen.add(id);
      const pos = drag?.[id] ?? { x: node.pos[0], y: node.pos[1] };
      const isSelected = selected.has(id);
      const size = sizes?.get(id);
      const box = boxes?.[id];
      const cached = this.nodeCache.get(id);
      if (
        cached &&
        cached.source === node &&
        cached.selected === isSelected &&
        cached.x === pos.x &&
        cached.y === pos.y &&
        cached.size === size &&
        cached.box === box
      ) {
        next.push(cached.rf);
        continue;
      }
      changed = true;
      let data = this.dataCache.get(id);
      if (!data) {
        data = { nodeId: id };
        this.dataCache.set(id, data);
      }
      const rf: RfNode = {
        id,
        type: "flowNode",
        position: pos,
        data,
        selected: isSelected,
        // The whole card drags; its controls opt out with `nodrag`.
        ...(box || (node.size && !node.collapsed)
          ? {
              style: box ?? { width: node.size?.[0], height: node.size?.[1] },
            }
          : {}),
        ...(size ? { measured: size } : {}),
      };
      this.nodeCache.set(id, {
        source: node,
        selected: isSelected,
        x: pos.x,
        y: pos.y,
        size,
        box,
        rf,
      });
      next.push(rf);
    }
    if (
      !changed &&
      next.length === this.lastNodes.length &&
      next.every((n, i) => n === this.lastNodes[i])
    ) {
      return this.lastNodes;
    }
    for (const key of this.nodeCache.keys())
      if (!seen.has(key)) {
        this.nodeCache.delete(key);
        this.dataCache.delete(key);
      }
    for (const key of this.groupCache.keys())
      if (!seen.has(GROUP_PREFIX + key)) this.groupCache.delete(key);
    for (const key of this.noteCache.keys())
      if (!seen.has(NOTE_PREFIX + key)) this.noteCache.delete(key);
    this.lastNodes = next;
    return next;
  }

  edges(
    graph: DocGraph,
    selected: ReadonlySet<string>,
    portTypeOf: (edge: FlowEdge) => string,
  ): RfEdge[] {
    const next: RfEdge[] = [];
    let changed = false;
    for (const id of graph.edgeOrder) {
      const edge = graph.edges[id];
      const isSelected = selected.has(id);
      const cached = this.edgeCache.get(id);
      if (cached && cached.source === edge && cached.selected === isSelected) {
        next.push(cached.rf);
        continue;
      }
      changed = true;
      const portType =
        cached && cached.source === edge ? cached.portType : portTypeOf(edge);
      const handles = edgeHandles(edge);
      const rf: RfEdge = {
        id,
        source: edge.source,
        target: edge.target,
        sourceHandle: handles.sourceHandle,
        targetHandle: handles.targetHandle,
        type: "flowEdge",
        selected: isSelected,
        data: { kind: edge.kind, portType },
      };
      this.edgeCache.set(id, {
        source: edge,
        selected: isSelected,
        portType,
        rf,
      });
      next.push(rf);
    }
    if (
      !changed &&
      next.length === this.lastEdges.length &&
      next.every((e, i) => e === this.lastEdges[i])
    ) {
      return this.lastEdges;
    }
    const live = new Set(graph.edgeOrder);
    for (const key of this.edgeCache.keys())
      if (!live.has(key)) this.edgeCache.delete(key);
    this.lastEdges = next;
    return next;
  }
}
