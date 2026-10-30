// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Edge, Node } from "@xyflow/react";
import { type DocGraph, GROUP_KEY, NOTE_KEY } from "../stores/doc-store";
import type { FlowReroute, RerouteWire } from "../stores/reroute-graph";
import type { FlowEdge, FlowGroup, FlowNode, FlowNote } from "../types";
import { edgeHandles } from "./connection-rules";
import { wireHandles } from "./reroutes";

export type FlowNodeData = { nodeId: string };
export type FlowEdgeData = {
  kind: "data" | "flow";
  portType: string;
  /** Cards this drawn segment belongs to, through any reroutes, for hover emphasis. */
  ends?: string[];
  /** A wire pinned to one style, whatever the toolbar says. */
  wireStyle?: "bus" | "curved";
};

export type RfNode = Node<
  | FlowNodeData
  | { groupId: string }
  | { noteId: string }
  | { rerouteId: string }
>;

export interface WireInfo {
  portType: string;
  ends: string[];
}
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

// Below a card even when selected: React Flow lifts a selected node by 1000.
export const GROUP_Z = -2000;

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
  private rerouteCache = new Map<
    string,
    { source: FlowReroute; selected: boolean; x: number; y: number; rf: RfNode }
  >();
  private edgeCache = new Map<
    string,
    { source: FlowEdge; selected: boolean; portType: string; rf: RfEdge }
  >();
  private wireCache = new Map<
    string,
    { source: RerouteWire; selected: boolean; key: string; rf: RfEdge }
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
        // Only the header and resize handles take the pointer; the body pans, clicks and box-selects like bare canvas.
        style: {
          ...(box ?? { width: group.rect[2], height: group.rect[3] }),
          pointerEvents: "none",
        },
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
    for (const id of graph.rerouteOrder) {
      const reroute = graph.reroutes[id];
      seen.add(id);
      const pos = drag?.[id] ?? { x: reroute.pos[0], y: reroute.pos[1] };
      const isSelected = selected.has(id);
      const cached = this.rerouteCache.get(id);
      if (
        cached &&
        cached.source === reroute &&
        cached.selected === isSelected &&
        cached.x === pos.x &&
        cached.y === pos.y
      ) {
        next.push(cached.rf);
        continue;
      }
      changed = true;
      const rf: RfNode = {
        id,
        type: "flowReroute",
        position: pos,
        data: { rerouteId: id },
        selected: isSelected,
      };
      this.rerouteCache.set(id, {
        source: reroute,
        selected: isSelected,
        x: pos.x,
        y: pos.y,
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
    for (const key of this.rerouteCache.keys())
      if (!seen.has(key)) this.rerouteCache.delete(key);
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
    wireInfo?: (wire: RerouteWire) => WireInfo,
  ): RfEdge[] {
    const next: RfEdge[] = [];
    let changed = false;
    for (const id of graph.edgeOrder) {
      // Drawn as its segments below.
      if (graph.wires[id]) continue;
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
    for (const id of graph.wireOrder) {
      const wire = graph.wires[id];
      const isSelected = selected.has(id);
      const info = wireInfo?.(wire) ?? {
        portType: "any",
        ends: [wire.source, wire.target],
      };
      const key = `${info.portType}|${info.ends.join(",")}`;
      const cached = this.wireCache.get(id);
      if (
        cached &&
        cached.source === wire &&
        cached.selected === isSelected &&
        cached.key === key
      ) {
        next.push(cached.rf);
        continue;
      }
      changed = true;
      const rf: RfEdge = {
        id,
        source: wire.source,
        target: wire.target,
        ...wireHandles(graph, wire),
        type: "flowEdge",
        selected: isSelected,
        // A segment through a reroute moves by moving the reroute, not its end.
        reconnectable: false,
        data: {
          kind: "data",
          portType: info.portType,
          ends: info.ends,
          ...(wire.style ? { wireStyle: wire.style } : {}),
        },
      };
      this.wireCache.set(id, { source: wire, selected: isSelected, key, rf });
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
    for (const key of this.wireCache.keys())
      if (!graph.wires[key]) this.wireCache.delete(key);
    this.lastEdges = next;
    return next;
  }
}
