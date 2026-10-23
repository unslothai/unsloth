// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Edge, Node, XYPosition } from "@xyflow/react";
import type {
  RecipeExecutionRecord,
  RecipeExecutionStatus,
} from "../../execution-types";
import { DEFAULT_NODE_HEIGHT, DEFAULT_NODE_WIDTH } from "../../constants";
import type { LayoutDirection, NodeConfig, RecipeNode } from "../../types";
import { getLayoutedElements } from "../layout";
import { readNodeHeight, readNodeWidth } from "../rf-node-dimensions";

// A super card is a view over the flat graph: members stay real nodes in the
// store, so the run payload never sees the grouping.
export type SuperCard = {
  id: string;
  title: string;
  nodeIds: string[];
  position: XYPosition;
};

export type SuperCardPort = {
  handleId: string;
  label: string;
};

export type SuperCardLastRun = {
  status: RecipeExecutionStatus;
  kind: RecipeExecutionRecord["kind"];
  doneColumns: number;
  totalColumns: number;
  finishedAt: number | null;
};

export type SuperCardNodeData = {
  kind: "super-card";
  cardId: string;
  title: string;
  memberCount: number;
  inputs: SuperCardPort[];
  outputs: SuperCardPort[];
  produces: string[];
  layoutDirection: LayoutDirection;
  runtimeState: "idle" | "running" | "done";
  lastRun: SuperCardLastRun | null;
  executionLocked: boolean;
};

export type UiSuperCard = {
  id: string;
  title: string;
  members: string[];
  x: number;
  y: number;
};

export const SUPER_CARD_NODE_TYPE = "super";
export const SUPER_CARD_WIDTH = 360;
const SUPER_CARD_HEIGHT_ESTIMATE = 380;

const COLUMN_KINDS: ReadonlySet<NodeConfig["kind"]> = new Set([
  "sampler",
  "llm",
  "expression",
  "validator",
]);

export function superCardInputHandle(externalId: string): string {
  return `sc-in:${externalId}`;
}

export function superCardOutputHandle(memberId: string): string {
  return `sc-out:${memberId}`;
}

export function buildSuperCardMembership(
  superCards: Record<string, SuperCard>,
): Map<string, string> {
  const owner = new Map<string, string>();
  for (const card of Object.values(superCards)) {
    for (const nodeId of card.nodeIds) {
      owner.set(nodeId, card.id);
    }
  }
  return owner;
}

export function getCombinableSelection(
  nodes: RecipeNode[],
  superCards: Record<string, SuperCard>,
): string[] {
  const owner = buildSuperCardMembership(superCards);
  return nodes
    .filter((node) => node.selected && !owner.has(node.id))
    .map((node) => node.id);
}

function nextSuperCardId(superCards: Record<string, SuperCard>): string {
  let max = 0;
  for (const id of Object.keys(superCards)) {
    const match = /^sc(\d+)$/.exec(id);
    if (match) {
      max = Math.max(max, Number(match[1]));
    }
  }
  return `sc${max + 1}`;
}

type Rect = { x: number; y: number; width: number; height: number };

const MEMBER_GAP = 16;

function memberRect(node: RecipeNode): Rect {
  return {
    x: node.position.x,
    y: node.position.y,
    width: readNodeWidth(node) ?? DEFAULT_NODE_WIDTH,
    height: readNodeHeight(node) ?? DEFAULT_NODE_HEIGHT,
  };
}

function superCardRect(card: SuperCard): Rect {
  return {
    ...card.position,
    width: SUPER_CARD_WIDTH,
    height: SUPER_CARD_HEIGHT_ESTIMATE,
  };
}

function rectsOverlap(a: Rect, b: Rect): boolean {
  return (
    a.x < b.x + b.width + MEMBER_GAP &&
    b.x < a.x + a.width + MEMBER_GAP &&
    a.y < b.y + b.height + MEMBER_GAP &&
    b.y < a.y + a.height + MEMBER_GAP
  );
}

export function membersOverlap(nodes: RecipeNode[]): boolean {
  const rects = nodes.map(memberRect);
  return rects.some((a, i) =>
    rects.slice(i + 1).some((b) => rectsOverlap(a, b)),
  );
}

// Members are hidden once grouped, so their spots are free: take the first
// one where the card clears every block still on the canvas.
function placeSuperCard(members: RecipeNode[], others: Rect[]): XYPosition {
  const corner = {
    x: Math.min(...members.map((node) => node.position.x)),
    y: Math.min(...members.map((node) => node.position.y)),
  };
  const candidates = [
    corner,
    ...[...members]
      .sort(
        (a, b) => a.position.y - b.position.y || a.position.x - b.position.x,
      )
      .map((node) => node.position),
  ];
  const free = candidates.find((point) => {
    const rect = {
      ...point,
      width: SUPER_CARD_WIDTH,
      height: SUPER_CARD_HEIGHT_ESTIMATE,
    };
    return others.every((other) => !rectsOverlap(rect, other));
  });
  return { ...(free ?? corner) };
}

// True when ungrouped members would land on a block or card still showing.
export function ungroupCollides(
  card: SuperCard,
  nodes: RecipeNode[],
  superCards: Record<string, SuperCard>,
): boolean {
  const members = new Set(card.nodeIds);
  const owner = buildSuperCardMembership(superCards);
  const memberRects = nodes
    .filter((node) => members.has(node.id))
    .map(memberRect);
  const others = [
    ...nodes
      .filter((node) => !(members.has(node.id) || owner.has(node.id)))
      .map(memberRect),
    ...Object.values(superCards)
      .filter((other) => other.id !== card.id)
      .map(superCardRect),
  ];
  return memberRects.some((a) => others.some((b) => rectsOverlap(a, b)));
}

export function estimateSuperCardHeight(data: SuperCardNodeData): number {
  const ports = Math.max(data.inputs.length, data.outputs.length, 1);
  const produceRows = Math.ceil(Math.min(data.produces.length, 5) / 2);
  return 64 + 32 + ports * 30 + 24 + produceRows * 34 + 32 + 56;
}

export function createSuperCard(
  superCards: Record<string, SuperCard>,
  nodes: RecipeNode[],
  nodeIds: string[],
): { superCards: Record<string, SuperCard>; card: SuperCard } | null {
  const owner = buildSuperCardMembership(superCards);
  const members = nodes.filter(
    (node) => nodeIds.includes(node.id) && !owner.has(node.id),
  );
  if (members.length < 2) {
    return null;
  }
  const id = nextSuperCardId(superCards);
  const memberIds = new Set(members.map((node) => node.id));
  const others = [
    ...nodes
      .filter((node) => !(memberIds.has(node.id) || owner.has(node.id)))
      .map(memberRect),
    ...Object.values(superCards).map(superCardRect),
  ];
  const card: SuperCard = {
    id,
    title: `Super card ${id.slice(2)}`,
    nodeIds: members.map((node) => node.id),
    position: placeSuperCard(members, others),
  };
  return { superCards: { ...superCards, [id]: card }, card };
}

export type SuperCardState = {
  nodes: RecipeNode[];
  superCards: Record<string, SuperCard>;
  openSuperCardId: string | null;
  selectedSuperCardIds: string[];
};

export function combineSelection(
  state: SuperCardState,
): Partial<SuperCardState> | null {
  if (state.openSuperCardId) {
    return null;
  }
  const created = createSuperCard(
    state.superCards,
    state.nodes,
    getCombinableSelection(state.nodes, state.superCards),
  );
  if (!created) {
    return null;
  }
  return {
    superCards: created.superCards,
    selectedSuperCardIds: [created.card.id],
    nodes: state.nodes.map((node) =>
      node.selected ? { ...node, selected: false } : node,
    ),
  };
}

// Members ride along with a dragged card so ungroup brings them back beside it.
export function moveSuperCard(
  state: Pick<SuperCardState, "nodes" | "superCards">,
  id: string,
  position: XYPosition,
): Pick<SuperCardState, "nodes" | "superCards"> | null {
  const card = state.superCards[id];
  if (
    !card ||
    (card.position.x === position.x && card.position.y === position.y)
  ) {
    return null;
  }
  const dx = position.x - card.position.x;
  const dy = position.y - card.position.y;
  const members = new Set(card.nodeIds);
  return {
    superCards: { ...state.superCards, [id]: { ...card, position } },
    nodes: state.nodes.map((node) =>
      members.has(node.id)
        ? {
            ...node,
            position: { x: node.position.x + dx, y: node.position.y + dy },
          }
        : node,
    ),
  };
}

// Members were never edited, so ungrouping only drops the card. Nothing is
// left selected, so the Combine action doesn't pop straight back up.
export function ungroupCard(
  state: SuperCardState,
  id: string,
): Partial<SuperCardState> | null {
  const card = state.superCards[id];
  if (!card) {
    return null;
  }
  return {
    superCards: removeSuperCard(state.superCards, id),
    openSuperCardId:
      state.openSuperCardId === id ? null : state.openSuperCardId,
    selectedSuperCardIds: state.selectedSuperCardIds.filter(
      (cardId) => cardId !== id,
    ),
    nodes: state.nodes.map((node) =>
      node.selected ? { ...node, selected: false } : node,
    ),
  };
}

export function removeSuperCard(
  superCards: Record<string, SuperCard>,
  id: string,
): Record<string, SuperCard> {
  if (!superCards[id]) {
    return superCards;
  }
  const { [id]: _removed, ...rest } = superCards;
  return rest;
}

export function pruneSuperCards(
  superCards: Record<string, SuperCard>,
  existingNodeIds: Set<string>,
): Record<string, SuperCard> {
  let changed = false;
  const next: Record<string, SuperCard> = {};
  for (const card of Object.values(superCards)) {
    const nodeIds = card.nodeIds.filter((id) => existingNodeIds.has(id));
    if (nodeIds.length !== card.nodeIds.length) {
      changed = true;
    }
    if (nodeIds.length === 0) {
      continue;
    }
    next[card.id] =
      nodeIds.length === card.nodeIds.length ? card : { ...card, nodeIds };
  }
  return changed ? next : superCards;
}

function nameOf(configs: Record<string, NodeConfig>, id: string): string {
  return configs[id]?.name ?? id;
}

export function summarizeSuperCard(
  card: SuperCard,
  edges: Edge[],
  configs: Record<string, NodeConfig>,
): { inputs: SuperCardPort[]; outputs: SuperCardPort[]; produces: string[] } {
  const members = new Set(card.nodeIds);
  const inputs = new Map<string, SuperCardPort>();
  const outputs = new Map<string, SuperCardPort>();
  for (const edge of edges) {
    const sourceIn = members.has(edge.source);
    const targetIn = members.has(edge.target);
    if (sourceIn === targetIn) {
      continue;
    }
    if (targetIn) {
      inputs.set(edge.source, {
        handleId: superCardInputHandle(edge.source),
        label: nameOf(configs, edge.source),
      });
    } else {
      outputs.set(edge.source, {
        handleId: superCardOutputHandle(edge.source),
        label: nameOf(configs, edge.source),
      });
    }
  }
  const produces = card.nodeIds.flatMap((id) => {
    const config = configs[id];
    return config && COLUMN_KINDS.has(config.kind) ? [config.name] : [];
  });
  return {
    inputs: [...inputs.values()],
    outputs: [...outputs.values()],
    produces,
  };
}

export function summarizeLastRun(
  card: SuperCard,
  configs: Record<string, NodeConfig>,
  executions: RecipeExecutionRecord[],
): SuperCardLastRun | null {
  const latest = executions.reduce<RecipeExecutionRecord | null>(
    (best, execution) =>
      !best || execution.createdAt > best.createdAt ? execution : best,
    null,
  );
  if (!latest) {
    return null;
  }
  const columns = card.nodeIds.flatMap((id) => {
    const config = configs[id];
    return config && COLUMN_KINDS.has(config.kind) ? [config.name] : [];
  });
  const completed = new Set(latest.completed_columns ?? []);
  const doneColumns =
    latest.status === "completed"
      ? columns.length
      : columns.filter((name) => completed.has(name)).length;
  return {
    status: latest.status,
    kind: latest.kind,
    doneColumns,
    totalColumns: columns.length,
    finishedAt: latest.finishedAt,
  };
}

// The card plus whatever it wires to, as shown on the root canvas.
export function superCardFocusIds(
  cardId: string,
  superCards: Record<string, SuperCard>,
  edges: Edge[],
): string[] {
  const card = superCards[cardId];
  if (!card) {
    return [];
  }
  const owner = buildSuperCardMembership(superCards);
  const members = new Set(card.nodeIds);
  const ids = new Set([cardId]);
  for (const edge of edges) {
    const sourceIn = members.has(edge.source);
    if (sourceIn === members.has(edge.target)) {
      continue;
    }
    const other = sourceIn ? edge.target : edge.source;
    ids.add(owner.get(other) ?? other);
  }
  return [...ids];
}

// Auto layout for the inside view: only runs when members would overlap, and
// keeps the group anchored where it was so the root canvas doesn't jump.
export function layoutSuperCardMembers(
  card: SuperCard,
  nodes: RecipeNode[],
  edges: Edge[],
  configs: Record<string, NodeConfig>,
  direction: LayoutDirection,
  force = false,
): RecipeNode[] | null {
  const members = new Set(card.nodeIds);
  const memberNodes = nodes.filter((node) => members.has(node.id));
  if (memberNodes.length < 2 || !(force || membersOverlap(memberNodes))) {
    return null;
  }
  const topBottom = direction === "TB";
  const laid = getLayoutedElements(
    memberNodes,
    edges.filter(
      (edge) => members.has(edge.source) && members.has(edge.target),
    ),
    {
      direction,
      nodesep: topBottom ? 120 : 80,
      ranksep: topBottom ? 140 : 80,
      configs,
    },
  ).nodes;
  const minOf = (list: RecipeNode[], axis: "x" | "y") =>
    Math.min(...list.map((node) => node.position[axis]));
  const dx = minOf(memberNodes, "x") - minOf(laid, "x");
  const dy = minOf(memberNodes, "y") - minOf(laid, "y");
  const positions = new Map(
    laid.map((node) => [
      node.id,
      { x: node.position.x + dx, y: node.position.y + dy },
    ]),
  );
  return nodes.map((node) => {
    const position = positions.get(node.id);
    return position ? { ...node, position } : node;
  });
}

function isAuxOf(nodeId: string, memberIds: Set<string>): boolean {
  if (!nodeId.startsWith("aux-")) {
    return false;
  }
  for (const id of memberIds) {
    if (nodeId.startsWith(`aux-${id}-`)) {
      return true;
    }
  }
  return false;
}

type SuperCardViewInput<N extends Node, E extends Edge> = {
  graph: { nodes: N[]; edges: E[] };
  superCards: Record<string, SuperCard>;
  openSuperCardId: string | null;
  selectedSuperCardIds: ReadonlySet<string>;
  configs: Record<string, NodeConfig>;
  edges: Edge[];
  layoutDirection: LayoutDirection;
  executions?: RecipeExecutionRecord[];
  runtime?: {
    runningNodeId: string | null;
    doneNodeIds: Set<string>;
    executionLocked: boolean;
  };
};

// Root view: members fold into their card and crossing wires re-attach to the
// card's handles. Inside view: only the open card's members and wires.
export function applySuperCardView<N extends Node, E extends Edge>({
  graph,
  superCards,
  openSuperCardId,
  selectedSuperCardIds,
  configs,
  edges,
  layoutDirection,
  executions = [],
  runtime,
}: SuperCardViewInput<N, E>): { nodes: N[]; edges: E[] } {
  const openCard = openSuperCardId ? superCards[openSuperCardId] : null;
  if (openCard) {
    const members = new Set(openCard.nodeIds);
    const visible = (id: string) => members.has(id) || isAuxOf(id, members);
    return {
      nodes: graph.nodes.filter((node) => visible(node.id)),
      edges: graph.edges.filter(
        (edge) => visible(edge.source) && visible(edge.target),
      ),
    };
  }
  const cards = Object.values(superCards);
  if (cards.length === 0) {
    return graph;
  }
  const owner = buildSuperCardMembership(superCards);
  const allMembers = new Set(owner.keys());
  const hidden = (id: string) => owner.has(id) || isAuxOf(id, allMembers);

  const cardNodes = cards.map((card) => {
    const summary = summarizeSuperCard(card, edges, configs);
    const running =
      Boolean(runtime?.runningNodeId) &&
      card.nodeIds.includes(runtime?.runningNodeId ?? "");
    const done =
      !running &&
      card.nodeIds.length > 0 &&
      card.nodeIds.every((id) => runtime?.doneNodeIds.has(id));
    const data: SuperCardNodeData = {
      kind: "super-card",
      cardId: card.id,
      title: card.title,
      memberCount: card.nodeIds.length,
      inputs: summary.inputs,
      outputs: summary.outputs,
      produces: summary.produces,
      layoutDirection,
      runtimeState: running ? "running" : done ? "done" : "idle",
      lastRun: summarizeLastRun(card, configs, executions),
      executionLocked: runtime?.executionLocked ?? false,
    };
    const node: Node<SuperCardNodeData> = {
      id: card.id,
      type: SUPER_CARD_NODE_TYPE,
      position: card.position,
      data,
      selected: selectedSuperCardIds.has(card.id),
      deletable: false,
      style: { width: SUPER_CARD_WIDTH },
    };
    // The canvas renders card nodes with their own node type.
    return node as unknown as N;
  });

  const seen = new Set<string>();
  const nextEdges: E[] = [];
  for (const edge of graph.edges) {
    const sourceCard = owner.get(edge.source);
    const targetCard = owner.get(edge.target);
    if (!(sourceCard || targetCard)) {
      if (!(hidden(edge.source) || hidden(edge.target))) {
        nextEdges.push(edge);
      }
      continue;
    }
    if (sourceCard && sourceCard === targetCard) {
      continue;
    }
    if (hidden(edge.source) && !sourceCard) {
      continue;
    }
    const source = sourceCard ?? edge.source;
    const target = targetCard ?? edge.target;
    const sourceHandle = sourceCard
      ? superCardOutputHandle(edge.source)
      : edge.sourceHandle;
    const targetHandle = targetCard
      ? superCardInputHandle(edge.source)
      : edge.targetHandle;
    const id = `sc-edge:${source}:${sourceHandle ?? ""}->${target}:${targetHandle ?? ""}`;
    if (seen.has(id)) {
      continue;
    }
    seen.add(id);
    nextEdges.push({
      ...edge,
      id,
      source,
      target,
      sourceHandle,
      targetHandle,
      selectable: false,
      deletable: false,
      focusable: false,
    });
  }

  return {
    nodes: [...graph.nodes.filter((node) => !hidden(node.id)), ...cardNodes],
    edges: nextEdges,
  };
}

export function superCardsToUi(
  superCards: Record<string, SuperCard>,
  configs: Record<string, NodeConfig>,
): UiSuperCard[] {
  return Object.values(superCards).flatMap((card) => {
    const members = card.nodeIds.flatMap((id) =>
      configs[id] ? [configs[id].name] : [],
    );
    if (members.length === 0) {
      return [];
    }
    return [
      {
        id: card.id,
        title: card.title,
        members,
        x: card.position.x,
        y: card.position.y,
      },
    ];
  });
}

export function superCardsFromUi(
  raw: unknown,
  nameToId: Map<string, string>,
): Record<string, SuperCard> {
  if (!Array.isArray(raw)) {
    return {};
  }
  const out: Record<string, SuperCard> = {};
  const taken = new Set<string>();
  for (const item of raw) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const record = item as Record<string, unknown>;
    const id = typeof record.id === "string" ? record.id.trim() : "";
    if (!id || out[id] || !Array.isArray(record.members)) {
      continue;
    }
    // One nesting level: a node belongs to at most one card.
    const nodeIds = record.members.flatMap((name) => {
      const nodeId = typeof name === "string" ? nameToId.get(name) : undefined;
      if (!nodeId || taken.has(nodeId)) {
        return [];
      }
      taken.add(nodeId);
      return [nodeId];
    });
    if (nodeIds.length === 0) {
      continue;
    }
    out[id] = {
      id,
      title:
        typeof record.title === "string" && record.title.trim()
          ? record.title
          : `Super card ${id}`,
      nodeIds,
      position: {
        x: typeof record.x === "number" ? record.x : 0,
        y: typeof record.y === "number" ? record.y : 0,
      },
    };
  }
  return out;
}
