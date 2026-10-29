// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DocGraph, DocMeta } from "../stores/doc-store";
import type { NodeSpec, NodeStatus } from "../types";
import { parseHandle } from "./connection-rules";
import {
  GROUP_PREFIX,
  NOTE_PREFIX,
  type RfEdge,
  type RfNode,
} from "./derive-rf";
import type { LayoutRequest, LayoutResult } from "./layout-core";
import type { PortRowSpec } from "./node-card";

// A super card is a view over the flat graph: its blocks stay real nodes with their own
// positions, so the document the engine compiles is the same combined, open or ungrouped.
export const SUPER_CARDS_KEY = "super_cards";
export const SUPER_PREFIX = "super:";

export interface SuperCard {
  id: string;
  title: string;
  members: string[];
  open?: boolean;
}

export interface SuperCardData {
  cardId: string;
  title: string;
  /** The colour of a group folded into the card, if any. */
  color?: string;
  memberIds: string[];
  /** Frames, notes and reroutes folded in with the blocks, where they sit; they move with the card. */
  folded: { id: string; x: number; y: number }[];
  inputs: PortRowSpec[];
  outputs: PortRowSpec[];
  [key: string]: unknown;
}

type Editor = Record<string, unknown> | undefined;

const DEFAULT_SIZE = { width: 260, height: 140 };

/** Every saved card whose blocks still exist; a block belongs to one card (one nesting level). */
export function readSuperCards(
  editor: Editor,
  graph: Pick<DocGraph, "nodes" | "nodeOrder" | "groups">,
): SuperCard[] {
  const raw = editor?.[SUPER_CARDS_KEY];
  if (!Array.isArray(raw)) return [];
  const taken = new Set<string>();
  const cards: SuperCard[] = [];
  for (const item of raw) {
    let card: SuperCard | null = null;
    if (typeof item === "string" && graph.groups[item]) {
      // Earlier builds of this branch saved a collapsed group's id.
      card = {
        id: item,
        title: superCardTitle(graph.groups[item]),
        members: graph.nodeOrder.filter((id) => graph.nodes[id].group === item),
      };
    } else if (item && typeof item === "object") {
      const r = item as Record<string, unknown>;
      if (typeof r.id === "string" && Array.isArray(r.members)) {
        card = {
          id: r.id,
          title:
            typeof r.title === "string" && r.title ? r.title : "Super card",
          members: r.members.filter((m): m is string => typeof m === "string"),
          ...(r.open === true ? { open: true } : {}),
        };
      }
    }
    if (!card || cards.some((c) => c.id === card.id)) continue;
    const members = card.members.filter(
      (id) => graph.nodes[id] && !taken.has(id),
    );
    if (!members.length) continue;
    for (const id of members) taken.add(id);
    cards.push({ ...card, members });
  }
  return cards;
}

export function writeSuperCards(
  editor: Editor,
  cards: SuperCard[],
): Record<string, unknown> {
  const { [SUPER_CARDS_KEY]: _old, ...rest } = editor ?? {};
  return cards.length ? { ...rest, [SUPER_CARDS_KEY]: cards } : rest;
}

export function superCardTitle(group: {
  label: string;
  kind?: unknown;
}): string {
  return group.kind === "recipe" ? `Recipe: ${group.label}` : group.label;
}

/** Selected blocks that can go into a new card: real blocks not already in one. */
export function combinableSelection(
  selected: readonly string[],
  graph: Pick<DocGraph, "nodes">,
  cards: SuperCard[],
): string[] {
  const taken = new Set(cards.flatMap((c) => c.members));
  return selected.filter((id) => graph.nodes[id] && !taken.has(id));
}

/** A whole group (a recipe, a For Each body) takes the group's name; anything else is numbered. */
function newTitle(
  graph: Pick<DocGraph, "nodes" | "nodeOrder" | "groups">,
  members: string[],
  cards: SuperCard[],
): string {
  const group = graph.nodes[members[0]]?.group;
  if (group && graph.groups[group]) {
    const inGroup = graph.nodeOrder.filter(
      (id) => graph.nodes[id].group === group,
    );
    const set = new Set(members);
    if (inGroup.length === members.length && inGroup.every((id) => set.has(id)))
      return superCardTitle(graph.groups[group]);
  }
  return `Super card ${cards.length + 1}`;
}

export function combineSuperCard(
  editor: Editor,
  graph: Pick<DocGraph, "nodes" | "nodeOrder" | "groups">,
  selected: readonly string[],
  id: string,
): { editor: Record<string, unknown>; card: SuperCard } | null {
  const cards = readSuperCards(editor, graph);
  const members = combinableSelection(selected, graph, cards);
  if (members.length < 2) return null;
  const card: SuperCard = {
    id,
    title: newTitle(graph, members, cards),
    members,
  };
  return { editor: writeSuperCards(editor, [...cards, card]), card };
}

export function updateSuperCard(
  editor: Editor,
  graph: Pick<DocGraph, "nodes" | "nodeOrder" | "groups">,
  id: string,
  patch: Partial<Pick<SuperCard, "open" | "title">> | null,
): Record<string, unknown> {
  const cards = readSuperCards(editor, graph);
  return writeSuperCards(
    editor,
    patch === null
      ? cards.filter((c) => c.id !== id)
      : cards.map((c) => {
          if (c.id !== id) return c;
          const { open: _open, ...rest } = { ...c, ...patch };
          return (patch.open ?? c.open) ? { ...rest, open: true } : rest;
        }),
  );
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
  return {
    label: `${name} · ${port?.label ?? parsed.port}`,
    color: port?.type,
  };
}

type Box = { x: number; y: number; right: number; bottom: number };

function boxOf(nodes: RfNode[]): Box | null {
  if (!nodes.length) return null;
  const w = (n: RfNode) => n.measured?.width ?? DEFAULT_SIZE.width;
  const h = (n: RfNode) => n.measured?.height ?? DEFAULT_SIZE.height;
  return {
    x: Math.min(...nodes.map((n) => n.position.x)),
    y: Math.min(...nodes.map((n) => n.position.y)),
    right: Math.max(...nodes.map((n) => n.position.x + w(n))),
    bottom: Math.max(...nodes.map((n) => n.position.y + h(n))),
  };
}

/** Closed cards become one node at their blocks' top-left; wires crossing in or out re-attach
 *  to the card's ports. */
export function applySuperCards(
  nodes: RfNode[],
  edges: RfEdge[],
  graph: DocGraph,
  cards: SuperCard[],
  specs: Record<string, NodeSpec>,
  portColors: Record<string, string> = {},
  selected: ReadonlySet<string> = new Set(),
): { nodes: RfNode[]; edges: RfEdge[] } {
  const closed = cards.filter((c) => !c.open);
  if (!closed.length) return { nodes, edges };
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const owner = new Map<string, string>();
  const boxes = new Map<string, Box>();
  for (const card of closed) {
    for (const id of card.members) owner.set(id, card.id);
    const box = boxOf(card.members.flatMap((id) => byId.get(id) ?? []));
    if (box) boxes.set(card.id, box);
  }
  // A group whose blocks are all in one card folds with it, so a combined recipe shows no
  // empty frame; notes and reroutes drawn among the blocks fold too.
  const groupCard = new Map<string, string | null>();
  for (const id of graph.nodeOrder) {
    const group = graph.nodes[id].group;
    if (!group) continue;
    const card = owner.get(id) ?? null;
    const seen = groupCard.get(group);
    groupCard.set(group, seen === undefined || seen === card ? card : null);
  }
  // Things inside a folded frame fold too (a recipe's notes sit below its blocks).
  // Folded frames first: a note in a frame belongs to that frame's card even where
  // another card's blocks overlap it.
  const areas: [string, Box][] = [];
  const colors = new Map<string, string>();
  for (const [group, card] of groupCard) {
    if (!card) continue;
    owner.set(GROUP_PREFIX + group, card);
    const color = graph.groups[group]?.color;
    if (color && !colors.has(card)) colors.set(card, color);
    const rect = graph.groups[group]?.rect;
    if (rect)
      areas.push([
        card,
        {
          x: rect[0],
          y: rect[1],
          right: rect[0] + rect[2],
          bottom: rect[1] + rect[3],
        },
      ]);
  }
  areas.push(...boxes);
  for (const n of nodes) {
    if (graph.nodes[n.id] || n.id.startsWith(GROUP_PREFIX)) continue;
    const x = n.position.x + (n.measured?.width ?? 0) / 2;
    const y = n.position.y + (n.measured?.height ?? 0) / 2;
    for (const [cardId, b] of areas) {
      if (x >= b.x && y >= b.y && x <= b.right && y <= b.bottom) {
        owner.set(n.id, cardId);
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
    cardId: string,
    nodeId: string,
    handle: string | null | undefined,
  ): string => {
    const handleId =
      side === "in"
        ? superCardInHandle(nodeId, handle ?? "")
        : superCardOutHandle(nodeId, handle ?? "");
    const table = side === "in" ? inputs : outputs;
    const ports = table.get(cardId) ?? new Map();
    table.set(cardId, ports);
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
    const source = sourceCard ? SUPER_PREFIX + sourceCard : edge.source;
    const target = targetCard ? SUPER_PREFIX + targetCard : edge.target;
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
  const cardNodes: RfNode[] = closed.flatMap((card) => {
    const box = boxes.get(card.id);
    if (!box) return [];
    const data: SuperCardData = {
      cardId: card.id,
      title: card.title,
      color: colors.get(card.id),
      memberIds: card.members,
      folded: nodes.flatMap((n) =>
        owner.get(n.id) === card.id && !graph.nodes[n.id]
          ? [{ id: n.id, x: n.position.x, y: n.position.y }]
          : [],
      ),
      inputs: sorted(inputs.get(card.id)),
      outputs: sorted(outputs.get(card.id)),
    };
    return [
      {
        id: SUPER_PREFIX + card.id,
        type: "flowSuperCard",
        position: { x: box.x, y: box.y },
        data: data as unknown as RfNode["data"],
        selected: selected.has(SUPER_PREFIX + card.id),
        deletable: false,
        dragHandle: ".flow-node-drag",
      },
    ];
  });
  return {
    nodes: [...nodes.filter((n) => !owner.has(n.id)), ...cardNodes],
    edges: nextEdges,
  };
}

/** Where everything folded into a closed card sits, so dragging the card carries it along. */
export function superCardDragMembers(
  node: RfNode,
  graph: Pick<DocGraph, "nodes">,
): Map<string, { x: number; y: number }> | null {
  if (node.type !== "flowSuperCard") return null;
  const { memberIds, folded } = node.data as unknown as SuperCardData;
  const members = new Map<string, { x: number; y: number }>();
  for (const id of memberIds) {
    const n = graph.nodes[id];
    if (n) members.set(id, { x: n.pos[0], y: n.pos[1] });
  }
  for (const { id, x, y } of folded) members.set(id, { x, y });
  return members;
}

const FINISHED: ReadonlySet<NodeStatus> = new Set([
  "done",
  "cached",
  "skipped",
]);

/** One status for the card from the steps that took part in the run (a recipe's columns never run on
 *  their own): a failure wins, then work in progress, then done when every one of them is. */
export function rollUpStatus(statuses: (NodeStatus | undefined)[]): {
  status: NodeStatus;
  done: number;
  total: number;
} {
  const list = statuses.filter((s): s is NodeStatus => !!s && s !== "idle");
  const done = list.filter((s) => FINISHED.has(s)).length;
  const has = (s: NodeStatus) => list.includes(s);
  let status: NodeStatus = "idle";
  if (has("failed")) status = "failed";
  else if (has("running") || has("loading_model")) status = "running";
  else if (has("queued") || has("waiting"))
    status = done ? "running" : "queued";
  else if (has("cancelled")) status = "cancelled";
  else if (
    list.length &&
    done === list.length &&
    !list.every((s) => s === "skipped")
  )
    status = "done";
  return { status, done, total: list.length };
}

/** Auto layout places a closed card as one block and moves its blocks by the same amount. */
export function foldLayout(
  request: LayoutRequest,
  state: { graph: DocGraph; meta: Pick<DocMeta, "editor"> },
  measured: (id: string) => { width?: number; height?: number } | undefined,
): { request: LayoutRequest; unfold: (result: LayoutResult) => LayoutResult } {
  const { graph } = state;
  const closed = readSuperCards(state.meta.editor, graph).filter(
    (c) => !c.open,
  );
  if (!closed.length) return { request, unfold: (result) => result };
  const owner = new Map<string, string>();
  for (const card of closed)
    for (const id of card.members) owner.set(id, SUPER_PREFIX + card.id);
  const blockOf = (id: string) => owner.get(id) ?? id;
  const nodes = request.nodes.filter((n) => !owner.has(n.id));
  for (const card of closed) {
    const size = measured(SUPER_PREFIX + card.id);
    nodes.push({
      id: SUPER_PREFIX + card.id,
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
    for (const card of closed) {
      const at = result[SUPER_PREFIX + card.id];
      delete next[SUPER_PREFIX + card.id];
      if (!at) continue;
      const x0 = Math.min(...card.members.map((id) => graph.nodes[id].pos[0]));
      const y0 = Math.min(...card.members.map((id) => graph.nodes[id].pos[1]));
      const dx = at[0] - x0;
      const dy = at[1] - y0;
      for (const id of card.members) {
        const [x, y] = graph.nodes[id].pos;
        next[id] = [x + dx, y + dy];
      }
      // Frames and notes folded into the card move with it, as they do when it is dragged.
      const members = new Set(card.members);
      const areas: [number, number, number, number][] = [];
      let right = x0;
      let bottom = y0;
      for (const id of card.members) {
        const size = measured(id);
        const [x, y] = graph.nodes[id].pos;
        right = Math.max(right, x + (size?.width ?? DEFAULT_SIZE.width));
        bottom = Math.max(bottom, y + (size?.height ?? DEFAULT_SIZE.height));
      }
      areas.push([x0, y0, right - x0, bottom - y0]);
      for (const groupId of graph.groupOrder) {
        const inGroup = graph.nodeOrder.filter(
          (id) => graph.nodes[id].group === groupId,
        );
        if (!inGroup.length || !inGroup.every((id) => members.has(id)))
          continue;
        const rect = graph.groups[groupId].rect;
        areas.push(rect);
        next[GROUP_PREFIX + groupId] = [rect[0] + dx, rect[1] + dy];
      }
      for (const noteId of graph.noteOrder) {
        const [x, y, w, h] = graph.notes[noteId].rect;
        const cx = x + w / 2;
        const cy = y + h / 2;
        if (
          areas.some(
            ([ax, ay, aw, ah]) =>
              cx >= ax && cy >= ay && cx <= ax + aw && cy <= ay + ah,
          )
        )
          next[NOTE_PREFIX + noteId] = [x + dx, y + dy];
      }
    }
    return next;
  };
  return { request: { ...request, nodes, edges }, unfold };
}

/** Cmd-click on a Mac, Ctrl-click elsewhere (React Flow's own default), plus Shift. */
export function multiSelectKeys(platform: string): string[] {
  return /mac|iphone|ipad/i.test(platform)
    ? ["Meta", "Shift"]
    : ["Control", "Shift"];
}
