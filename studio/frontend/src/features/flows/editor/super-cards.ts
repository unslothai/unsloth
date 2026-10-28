// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DocGraph, DocMeta } from "../stores/doc-store";
import type { NodeSpec, NodeStatus } from "../types";
import { parseHandle } from "./connection-rules";
import { GROUP_PREFIX, type RfEdge, type RfNode } from "./derive-rf";
import type { LayoutRequest, LayoutResult } from "./layout-core";
import type { PortRowSpec } from "./node-card";

// A super card is a collapsed group: a view over the flat graph. Members stay real nodes with
// their own positions, so the document the engine compiles is the same collapsed or open.
export const SUPER_CARDS_KEY = "super_cards";

export interface SuperCardData {
  groupId: string;
  memberIds: string[];
  inputs: PortRowSpec[];
  outputs: PortRowSpec[];
  [key: string]: unknown;
}

export function collapsedGroups(
  editor: Record<string, unknown> | undefined,
  graph: Pick<DocGraph, "groups">,
): Set<string> {
  const raw = editor?.[SUPER_CARDS_KEY];
  if (!Array.isArray(raw)) return new Set();
  return new Set(
    raw.filter(
      (id): id is string => typeof id === "string" && !!graph.groups[id],
    ),
  );
}

export function withSuperCard(
  editor: Record<string, unknown> | undefined,
  groupId: string,
  collapsed: boolean,
): Record<string, unknown> {
  const raw = editor?.[SUPER_CARDS_KEY];
  const current = Array.isArray(raw)
    ? raw.filter((id): id is string => typeof id === "string")
    : [];
  const next = collapsed
    ? current.includes(groupId)
      ? current
      : [...current, groupId]
    : current.filter((id) => id !== groupId);
  const { [SUPER_CARDS_KEY]: _old, ...rest } = editor ?? {};
  return next.length ? { ...rest, [SUPER_CARDS_KEY]: next } : rest;
}

export function superCardTitle(group: { label: string; kind?: unknown }): string {
  return group.kind === "recipe" ? `Recipe: ${group.label}` : group.label;
}

export function superCardInHandle(nodeId: string, handle: string): string {
  return `sc-i:${nodeId}:${handle}`;
}

export function superCardOutHandle(nodeId: string, handle: string): string {
  return `sc-o:${nodeId}:${handle}`;
}

function portLabel(
  graph: DocGraph,
  specs: Record<string, NodeSpec>,
  nodeId: string,
  handle: string | null | undefined,
  side: "in" | "out",
): { label: string; color?: string } {
  const node = graph.nodes[nodeId];
  const spec = node ? specs[node.type] : undefined;
  const name = node ? node.label || spec?.title || nodeId : "Wire";
  const parsed = parseHandle(handle);
  if (!parsed || parsed.kind === "flow")
    return { label: parsed?.side === "out" ? `${name} ›` : name };
  const ports = side === "in" ? spec?.inputs : spec?.outputs;
  const port = ports?.find((p) => p.id === parsed.port);
  return { label: `${name} · ${port?.label ?? parsed.port}`, color: port?.type };
}

/** Collapsed groups become one card; wires crossing the frame re-attach to the card's ports. */
export function applySuperCards(
  nodes: RfNode[],
  edges: RfEdge[],
  graph: DocGraph,
  collapsed: ReadonlySet<string>,
  specs: Record<string, NodeSpec>,
  portColors: Record<string, string> = {},
): { nodes: RfNode[]; edges: RfEdge[] } {
  if (!collapsed.size) return { nodes, edges };
  const owner = new Map<string, string>();
  const members = new Map<string, string[]>();
  for (const id of graph.nodeOrder) {
    const group = graph.nodes[id].group;
    if (!group || !collapsed.has(group)) continue;
    owner.set(id, group);
    members.set(group, [...(members.get(group) ?? []), id]);
  }
  // Anything else drawn inside a collapsed frame (notes, reroutes, a recipe's side cards) folds with it.
  for (const n of nodes) {
    if (graph.nodes[n.id] || n.id.startsWith(GROUP_PREFIX)) continue;
    const x = n.position.x + (n.measured?.width ?? 0) / 2;
    const y = n.position.y + (n.measured?.height ?? 0) / 2;
    for (const groupId of collapsed) {
      const [gx, gy, gw, gh] = graph.groups[groupId].rect;
      if (x >= gx && y >= gy && x <= gx + gw && y <= gy + gh) {
        owner.set(n.id, groupId);
        break;
      }
    }
  }
  const visible = new Set(
    nodes.filter((n) => !owner.has(n.id)).map((n) => n.id),
  );
  const inputs = new Map<string, Map<string, PortRowSpec & { y: number }>>();
  const outputs = new Map<string, Map<string, PortRowSpec & { y: number }>>();
  const addPort = (
    side: "in" | "out",
    group: string,
    nodeId: string,
    handle: string | null | undefined,
  ): string => {
    const handleId =
      side === "in"
        ? superCardInHandle(nodeId, handle ?? "")
        : superCardOutHandle(nodeId, handle ?? "");
    const table = side === "in" ? inputs : outputs;
    const ports = table.get(group) ?? new Map();
    table.set(group, ports);
    if (!ports.has(handleId)) {
      const { label, color } = portLabel(graph, specs, nodeId, handle, side);
      ports.set(handleId, {
        handleId,
        label,
        title: label,
        ...(color ? { color: portColors[color] ?? "#cbd5e1" } : {}),
        y: graph.nodes[nodeId]?.pos[1] ?? 0,
      });
    }
    return handleId;
  };

  const seen = new Set<string>();
  const nextEdges: RfEdge[] = [];
  for (const edge of edges) {
    const sourceCard = owner.get(edge.source);
    const targetCard = owner.get(edge.target);
    if (!(sourceCard || targetCard)) {
      if (visible.has(edge.source) && visible.has(edge.target))
        nextEdges.push(edge);
      continue;
    }
    if (sourceCard && sourceCard === targetCard) continue;
    if (!(sourceCard || visible.has(edge.source))) continue;
    if (!(targetCard || visible.has(edge.target))) continue;
    const sourceHandle = sourceCard
      ? addPort("out", sourceCard, edge.source, edge.sourceHandle)
      : edge.sourceHandle;
    const targetHandle = targetCard
      ? addPort("in", targetCard, edge.target, edge.targetHandle)
      : edge.targetHandle;
    const source = sourceCard ? GROUP_PREFIX + sourceCard : edge.source;
    const target = targetCard ? GROUP_PREFIX + targetCard : edge.target;
    const key = `${source}:${sourceHandle}->${target}:${targetHandle}`;
    if (seen.has(key)) continue;
    seen.add(key);
    nextEdges.push({
      ...edge,
      source,
      target,
      sourceHandle,
      targetHandle,
      selectable: false,
      deletable: false,
      focusable: false,
    });
  }

  const sorted = (ports?: Map<string, PortRowSpec & { y: number }>) =>
    [...(ports?.values() ?? [])]
      .sort((a, b) => a.y - b.y)
      .map(({ y: _y, ...port }) => port);
  const nextNodes: RfNode[] = [];
  for (const node of nodes) {
    if (owner.has(node.id)) continue;
    const groupId = node.id.startsWith(GROUP_PREFIX)
      ? node.id.slice(GROUP_PREFIX.length)
      : null;
    if (!(groupId && collapsed.has(groupId))) {
      nextNodes.push(node);
      continue;
    }
    const data: SuperCardData = {
      groupId,
      memberIds: members.get(groupId) ?? [],
      inputs: sorted(inputs.get(groupId)),
      outputs: sorted(outputs.get(groupId)),
    };
    nextNodes.push({
      id: node.id,
      type: "flowSuperCard",
      position: node.position,
      data,
      selected: node.selected,
      deletable: false,
      dragHandle: ".flow-node-drag",
    });
  }
  return { nodes: nextNodes, edges: nextEdges };
}

const FINISHED: ReadonlySet<NodeStatus> = new Set(["done", "cached", "skipped"]);

/** One status for the card: a failure wins, then work in progress, then done when every step is. */
export function rollUpStatus(statuses: (NodeStatus | undefined)[]): {
  status: NodeStatus;
  done: number;
} {
  const list = statuses.map((s) => s ?? "idle");
  const done = list.filter((s) => FINISHED.has(s)).length;
  const has = (s: NodeStatus) => list.includes(s);
  let status: NodeStatus = "idle";
  if (has("failed")) status = "failed";
  else if (has("running") || has("loading_model")) status = "running";
  else if (has("queued") || has("waiting")) status = done ? "running" : "queued";
  else if (list.length && done === list.length && !list.every((s) => s === "skipped"))
    status = "done";
  else if (has("cancelled")) status = "cancelled";
  return { status, done };
}

/** Auto layout places a collapsed card as one block and moves its members by the same amount. */
export function foldLayout(
  request: LayoutRequest,
  state: { graph: DocGraph; meta: Pick<DocMeta, "editor"> },
  measured: (id: string) => { width?: number; height?: number } | undefined,
): { request: LayoutRequest; unfold: (result: LayoutResult) => LayoutResult } {
  const { graph } = state;
  const collapsed = collapsedGroups(state.meta.editor, graph);
  if (!collapsed.size) return { request, unfold: (result) => result };
  const blockOf = (id: string) => {
    const group = graph.nodes[id]?.group;
    return group && collapsed.has(group) ? GROUP_PREFIX + group : id;
  };
  const nodes = request.nodes.filter((n) => blockOf(n.id) === n.id);
  for (const groupId of collapsed) {
    const size = measured(GROUP_PREFIX + groupId);
    nodes.push({
      id: GROUP_PREFIX + groupId,
      width: size?.width ?? 288,
      height: size?.height ?? 140,
    });
  }
  const seen = new Set<string>();
  const edges = request.edges.flatMap((edge) => {
    const source = blockOf(edge.source);
    const target = blockOf(edge.target);
    const key = `${source}->${target}`;
    if (source === target || seen.has(key)) return [];
    seen.add(key);
    return [{ ...edge, source, target }];
  });
  const unfold = (result: LayoutResult): LayoutResult => {
    const next = { ...result };
    for (const groupId of collapsed) {
      const at = result[GROUP_PREFIX + groupId];
      if (!at) continue;
      const rect = graph.groups[groupId].rect;
      const [dx, dy] = [at[0] - rect[0], at[1] - rect[1]];
      for (const id of graph.nodeOrder) {
        const node = graph.nodes[id];
        if (node.group === groupId)
          next[id] = [node.pos[0] + dx, node.pos[1] + dy];
      }
    }
    return next;
  };
  return { request: { ...request, nodes, edges }, unfold };
}
