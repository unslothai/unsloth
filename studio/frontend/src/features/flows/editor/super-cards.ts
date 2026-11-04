// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DocGraph, DocMeta } from "../stores/doc-store";
import { rerouteDownstream, rerouteUpstream } from "../stores/reroute-graph";
import type { NodeSpec, NodeStatus } from "../types";
import { inHandle, outHandle, parseHandle } from "./connection-rules";
import {
  GROUP_PREFIX,
  NOTE_PREFIX,
  type RfEdge,
  type RfNode,
} from "./derive-rf";
import type { LayoutRequest, LayoutResult } from "./layout-core";
import type { PortRowSpec } from "./node-card";
import { rerouteSize } from "./reroutes";
import {
  foldedGroups,
  readSuperCards,
  SUPER_CARDS_KEY,
  type SuperCard,
  superCardTitle,
} from "./super-card-records";

export {
  foldedGroups,
  readSuperCards,
  SUPER_CARDS_KEY,
  type SuperCard,
  superCardTitle,
} from "./super-card-records";

// A super card is a view over the flat graph: its blocks stay real nodes with their own
// positions, so the document the engine compiles is the same combined, open or ungrouped.
export const SUPER_PREFIX = "super:";

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

/** Ctrl+A: every block, plus each closed card, which is what the canvas shows and Delete acts on. */
export function selectAllIds(
  editor: Editor,
  graph: Pick<DocGraph, "nodes" | "nodeOrder" | "groups">,
): string[] {
  const closed = readSuperCards(editor, graph).filter((c) => !c.open);
  return [...graph.nodeOrder, ...closed.map((c) => SUPER_PREFIX + c.id)];
}

/** The selection with each closed super card replaced by the blocks in it, for copy and duplicate. */
export function selectedBlocks(
  selected: readonly string[],
  editor: Editor,
  graph: Pick<DocGraph, "nodes" | "nodeOrder" | "groups">,
): string[] {
  const cards = new Map(
    readSuperCards(editor, graph).map((c) => [SUPER_PREFIX + c.id, c.members]),
  );
  const out = new Set<string>();
  for (const id of selected)
    for (const m of cards.get(id) ?? [id]) if (graph.nodes[m]) out.add(m);
  return [...out];
}

export function writeSuperCards(
  editor: Editor,
  cards: SuperCard[],
): Record<string, unknown> {
  const { [SUPER_CARDS_KEY]: _old, ...rest } = editor ?? {};
  return cards.length ? { ...rest, [SUPER_CARDS_KEY]: cards } : rest;
}

/** Selected blocks not already in a card, plus closed cards, which a new card absorbs. */
export function combinableSelection(
  selected: readonly string[],
  graph: Pick<DocGraph, "nodes">,
  cards: SuperCard[],
): string[] {
  const taken = new Set(cards.flatMap((c) => c.members));
  const closed = new Set(
    cards.filter((c) => !c.open).map((c) => SUPER_PREFIX + c.id),
  );
  return selected.filter(
    (id) => (graph.nodes[id] && !taken.has(id)) || closed.has(id),
  );
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
  const picked = combinableSelection(selected, graph, cards);
  // Cards stay one level deep: a selected card's blocks move into the new one.
  const absorbed = cards.filter((c) => picked.includes(SUPER_PREFIX + c.id));
  const members = picked.flatMap(
    (pick) =>
      absorbed.find((c) => SUPER_PREFIX + c.id === pick)?.members ?? [pick],
  );
  if (picked.length < 2 || members.length < 2) return null;
  const kept = cards.filter((c) => !absorbed.includes(c));
  const card: SuperCard = {
    id,
    title: absorbed[0]?.title ?? newTitle(graph, members, kept),
    members,
  };
  return { editor: writeSuperCards(editor, [...kept, card]), card };
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
  // A closed card's blocks never mount, so after a reload their saved size (in style) stands in.
  const style = (n: RfNode) => (n.style ?? {}) as { width?: unknown; height?: unknown };
  const w = (n: RfNode) =>
    n.measured?.width ??
    (typeof style(n).width === "number" ? (style(n).width as number) : DEFAULT_SIZE.width);
  const h = (n: RfNode) =>
    n.measured?.height ??
    (typeof style(n).height === "number" ? (style(n).height as number) : DEFAULT_SIZE.height);
  return {
    x: Math.min(...nodes.map((n) => n.position.x)),
    y: Math.min(...nodes.map((n) => n.position.y)),
    right: Math.max(...nodes.map((n) => n.position.x + w(n))),
    bottom: Math.max(...nodes.map((n) => n.position.y + h(n))),
  };
}

/** Closed cards become one node at their blocks' top-left; wires crossing in or out re-attach
 *  to the card's ports. */
/** Each group whose blocks all sit in one closed card, with that card, in the order both the
 * canvas and Tidy up check their frames, so a note where two folded frames overlap has one owner. */
function foldedFrames(
  graph: DocGraph,
  owner: ReadonlyMap<string, string>,
): [string, string][] {
  const groupCard = new Map<string, string | null>();
  for (const id of graph.nodeOrder) {
    const group = graph.nodes[id].group;
    if (!group) continue;
    const card = owner.get(id) ?? null;
    const seen = groupCard.get(group);
    groupCard.set(group, seen === undefined || seen === card ? card : null);
  }
  return [...groupCard].flatMap(([group, card]): [string, string][] =>
    card ? [[group, card]] : [],
  );
}

type Measured = (id: string) => { width?: number; height?: number } | undefined;

/** Which closed card each note ("note:<id>") and reroute folds into: a card's saved list where it
 * has one, else by position (a folded frame first, then the card's blocks), from saved sizes. */
export function extraOwners(
  graph: DocGraph,
  closed: SuperCard[],
  measured: Measured = () => undefined,
): Map<string, string> {
  const found = new Map<string, string>();
  const owner = new Map<string, string>();
  for (const card of closed) for (const id of card.members) owner.set(id, card.id);
  for (const card of closed)
    for (const id of card.extras ?? []) if (!found.has(id)) found.set(id, card.id);
  const byPosition = new Set(closed.filter((c) => !c.extras).map((c) => c.id));
  if (!byPosition.size) return found;
  const areas: [string, [number, number, number, number]][] = [];
  for (const [group, card] of foldedFrames(graph, owner))
    if (byPosition.has(card) && graph.groups[group]) areas.push([card, graph.groups[group].rect]);
  for (const card of closed) {
    if (!byPosition.has(card.id)) continue;
    const members = card.members.filter((id) => graph.nodes[id]);
    if (!members.length) continue;
    const x0 = Math.min(...members.map((id) => graph.nodes[id].pos[0]));
    const y0 = Math.min(...members.map((id) => graph.nodes[id].pos[1]));
    let right = x0;
    let bottom = y0;
    for (const id of members) {
      const size = measured(id);
      const saved = graph.nodes[id].collapsed ? null : graph.nodes[id].size;
      const [x, y] = graph.nodes[id].pos;
      right = Math.max(right, x + (size?.width ?? saved?.[0] ?? DEFAULT_SIZE.width));
      bottom = Math.max(bottom, y + (size?.height ?? saved?.[1] ?? DEFAULT_SIZE.height));
    }
    areas.push([card.id, [x0, y0, right - x0, bottom - y0]]);
  }
  const place = (id: string, cx: number, cy: number) => {
    if (found.has(id)) return;
    const hit = areas.find(
      ([, [ax, ay, aw, ah]]) => cx >= ax && cy >= ay && cx <= ax + aw && cy <= ay + ah,
    );
    if (hit) found.set(id, hit[0]);
  };
  for (const noteId of graph.noteOrder) {
    const [x, y, w, h] = graph.notes[noteId].rect;
    place(NOTE_PREFIX + noteId, x + w / 2, y + h / 2);
  }
  for (const id of graph.rerouteOrder ?? []) {
    const reroute = graph.reroutes[id];
    const size = rerouteSize(reroute.dir);
    place(id, reroute.pos[0] + size.width / 2, reroute.pos[1] + size.height / 2);
  }
  return found;
}

/** The card list with ``id`` closed and its notes and reroutes recorded as they sit now. */
export function closeWithExtras(
  graph: DocGraph,
  cards: SuperCard[],
  id: string,
  measured: Measured = () => undefined,
): SuperCard[] {
  const closing: SuperCard[] = cards.map((c) => {
    if (c.id !== id) return c;
    const { open: _open, extras: _extras, ...rest } = c;
    return rest;
  });
  const owners = extraOwners(graph, closing.filter((c) => !c.open), measured);
  const extras = [...owners].flatMap(([extra, card]) => (card === id ? [extra] : []));
  return closing.map((c) => (c.id === id ? { ...c, extras } : c));
}

export function applySuperCards(
  nodes: RfNode[],
  edges: RfEdge[],
  graph: DocGraph,
  cards: SuperCard[],
  specs: Record<string, NodeSpec>,
  portColors: Record<string, string> = {},
  selected: ReadonlySet<string> = new Set(),
  measuredOf: (id: string) => { width: number; height: number } | undefined = () =>
    undefined,
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
  // Things inside a folded frame fold too (a recipe's notes sit below its blocks).
  // Folded frames first: a note in a frame belongs to that frame's card even where
  // another card's blocks overlap it.
  const areas: [string, Box][] = [];
  const colors = new Map<string, string>();
  // A card that recorded its notes and reroutes when it closed keeps exactly those.
  const listed = new Set<string>();
  for (const card of closed)
    for (const id of card.extras ?? [])
      if (!owner.has(id)) {
        owner.set(id, card.id);
        listed.add(id);
      }
  const byPosition = new Set(closed.filter((c) => !c.extras).map((c) => c.id));
  for (const [group, card] of foldedFrames(graph, owner)) {
    owner.set(GROUP_PREFIX + group, card);
    const color = graph.groups[group]?.color;
    if (color && !colors.has(card)) colors.set(card, color);
    const rect = graph.groups[group]?.rect;
    if (rect && byPosition.has(card))
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
  areas.push(...[...boxes].filter(([card]) => byPosition.has(card)));
  for (const n of nodes) {
    if (graph.nodes[n.id] || n.id.startsWith(GROUP_PREFIX) || listed.has(n.id)) continue;
    // A note not yet mounted has no measurement: its saved size is in its style.
    const style = n.style as { width?: unknown; height?: unknown } | undefined;
    const w = n.measured?.width ?? (typeof style?.width === "number" ? style.width : 0);
    const h = n.measured?.height ?? (typeof style?.height === "number" ? style.height : 0);
    const x = n.position.x + w / 2;
    const y = n.position.y + h / 2;
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

  // A reroute folded into a card stands for the block port it carries, so one output fanned out
  // through several reroutes is one port, labelled by its block; a loose reroute keeps its own.
  const portOf = (
    side: "in" | "out",
    cardId: string,
    nodeId: string,
    handle: string | null | undefined,
  ): [string, string | null | undefined] => {
    if (!graph.reroutes[nodeId]) return [nodeId, handle];
    if (side === "out") {
      const up = rerouteUpstream(graph, nodeId);
      return up && owner.get(up.node) === cardId
        ? [up.node, outHandle(up.port)]
        : [nodeId, handle];
    }
    const down = rerouteDownstream(graph, nodeId);
    return down.length === 1 && owner.get(down[0].node) === cardId
      ? [down[0].node, inHandle(down[0].port)]
      : [nodeId, handle];
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
      ? addPort("out", sourceCard, ...portOf("out", sourceCard, edge.source, edge.sourceHandle))
      : edge.sourceHandle;
    const targetHandle = targetCard
      ? addPort("in", targetCard, ...portOf("in", targetCard, edge.target, edge.targetHandle))
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
        dragHandle: ".flow-node-drag",
        // A card rebuilt on every drag or selection keeps its measurement, or React Flow hides it
        // to measure again and the card and its wires blink.
        ...(measuredOf(SUPER_PREFIX + card.id)
          ? { measured: measuredOf(SUPER_PREFIX + card.id) }
          : {}),
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
/** The graph Tidy up's request is built from: a closed card's blocks leave a group the card holds only
 *  part of, so their wires reach the card instead of the group's placeholder. */
export function tidyGraph(
  graph: DocGraph,
  editor: Record<string, unknown> | undefined,
): DocGraph {
  const folded = foldedGroups(editor, graph);
  let nodes: DocGraph["nodes"] | null = null;
  for (const card of readSuperCards(editor, graph)) {
    if (card.open) continue;
    for (const id of card.members) {
      const group = graph.nodes[id].group;
      if (!group || folded.has(group)) continue;
      nodes ??= { ...graph.nodes };
      nodes[id] = { ...graph.nodes[id], group: null };
    }
  }
  return nodes ? { ...graph, nodes } : graph;
}

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
  // Tidy up lays a group out as one block; a group wholly inside a card folds into it too.
  const frames = foldedFrames(graph, owner);
  for (const [groupId, card] of frames) owner.set(GROUP_PREFIX + groupId, card);
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
    const shift = new Map<string, [number, number]>();
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
      shift.set(SUPER_PREFIX + card.id, [dx, dy]);
    }
    // Frames and notes folded into a card move with it, as they do when it is dragged.
    for (const [groupId, card] of frames) {
      const d = shift.get(card);
      const rect = graph.groups[groupId]?.rect;
      if (!d || !rect) continue;
      next[GROUP_PREFIX + groupId] = [rect[0] + d[0], rect[1] + d[1]];
    }
    // Notes and reroutes folded in move the same way, with the card the canvas folds them into.
    for (const [id, card] of extraOwners(graph, closed, measured)) {
      const d = shift.get(SUPER_PREFIX + card);
      if (!d) continue;
      if (id.startsWith(NOTE_PREFIX)) {
        const note = graph.notes[id.slice(NOTE_PREFIX.length)];
        if (note) next[id] = [note.rect[0] + d[0], note.rect[1] + d[1]];
      } else if (graph.reroutes[id]) {
        const [x, y] = graph.reroutes[id].pos;
        next[id] = [x + d[0], y + d[1]];
      }
    }
    return next;
  };
  return { request: { ...request, nodes, edges }, unfold };
}

/** Whether a selection box (screen coords inside the pane) fully covers a flow-space rect. */
export function boxCovers(
  box: { x: number; y: number; width: number; height: number },
  [tx, ty, k]: [number, number, number],
  rect: { x: number; y: number; width: number; height: number },
): boolean {
  const x = (box.x - tx) / k;
  const y = (box.y - ty) / k;
  return (
    rect.x >= x &&
    rect.y >= y &&
    rect.x + rect.width <= x + box.width / k &&
    rect.y + rect.height <= y + box.height / k
  );
}

/** Cmd-click on a Mac, Ctrl-click elsewhere (React Flow's own default), plus Shift. */
export function multiSelectKeys(platform: string): string[] {
  return isApplePlatform(platform) ? ["Meta", "Shift"] : ["Control", "Shift"];
}

export function isApplePlatform(platform: string): boolean {
  return /mac|iphone|ipad/i.test(platform);
}

/** A Mac turns Ctrl-click into a right-click, so React Flow never sees it; add or drop the card here. */
export function toggledSelection(
  selected: readonly string[],
  id: string,
): string[] {
  return selected.includes(id)
    ? selected.filter((other) => other !== id)
    : [...selected, id];
}
