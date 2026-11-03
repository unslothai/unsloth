// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The saved super card records, with no editor imports, so the doc store can read them too.

import type { DocGraph } from "../stores/doc-store";

export const SUPER_CARDS_KEY = "super_cards";

export interface SuperCard {
  id: string;
  title: string;
  members: string[];
  open?: boolean;
  /** Notes ("note:<id>") and reroutes folded in when the card closed. Kept as a list, so a note
   * that later lands among the hidden blocks (Tidy up moving its own block) is not swallowed;
   * absent on cards saved before, which fold by position. */
  extras?: string[];
}

type Editor = Record<string, unknown> | undefined;

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
          ...(Array.isArray(r.extras)
            ? { extras: r.extras.filter((m): m is string => typeof m === "string") }
            : {}),
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

export function superCardTitle(group: {
  label: string;
  kind?: unknown;
}): string {
  return group.kind === "recipe" ? `Recipe: ${group.label}` : group.label;
}

/** Groups whose blocks are all in one closed card: their frame is hidden, so nothing new joins them. */
export function foldedGroups(
  editor: Editor,
  graph: Pick<DocGraph, "nodes" | "nodeOrder" | "groups">,
): Set<string> {
  const owner = new Map<string, string>();
  for (const card of readSuperCards(editor, graph))
    if (!card.open) for (const id of card.members) owner.set(id, card.id);
  const cardOf = new Map<string, string | null>();
  for (const id of graph.nodeOrder) {
    const group = graph.nodes[id].group;
    if (!group) continue;
    const card = owner.get(id) ?? null;
    const seen = cardOf.get(group);
    cardOf.set(group, seen === undefined || seen === card ? card : null);
  }
  return new Set([...cardOf].flatMap(([group, card]) => (card ? [group] : [])));
}
