// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The model / tool cards of every recipe group as canvas-only nodes and wires. They are not flow nodes:
// the recipe config is their source, and a dragged card keeps its spot in the group's `ui`.

import type { DocGraph } from "../stores/doc-store";
import { GROUP_KEY } from "../stores/doc-store";
import type { FlowGroup } from "../types";
import { groupGenerate, isRecipeGroup } from "./recipe-columns";
import { AUX_PREFIX, type AuxCard, recipeAux } from "./recipe-look";

export const AUX_UI_KEY = "flows_aux_positions";
const CARD_GAP = 24;
const CARD_HEIGHT = 84;

export interface AuxElement {
  card: AuxCard;
  x: number;
  y: number;
}

type Size = { width: number; height: number };
type Point = { x: number; y: number };

export function isAuxId(id: string): boolean {
  return id.startsWith(AUX_PREFIX);
}

function savedPositions(group: FlowGroup): Record<string, [number, number]> {
  const saved = group.ui?.[AUX_UI_KEY];
  return saved && typeof saved === "object"
    ? (saved as Record<string, [number, number]>)
    : {};
}

/** Where each card sits: its saved spot in the group, else stacked under the group's Recipe Dataset step. */
export function auxLayout(
  graph: DocGraph,
  drag: Record<string, Point> | null,
  sizes: ReadonlyMap<string, Size>,
): {
  elements: AuxElement[];
  links: { id: string; source: string; target: string }[];
  /** recipe groups whose frame must grow to hold their cards: group id -> height */
  heights: Record<string, number>;
} {
  const elements: AuxElement[] = [];
  const links: { id: string; source: string; target: string }[] = [];
  const heights: Record<string, number> = {};
  for (const groupId of graph.groupOrder) {
    const group = graph.groups[groupId];
    if (!isRecipeGroup(group)) continue;
    const { cards, links: groupLinks } = recipeAux(graph, group);
    if (!cards.length) continue;
    links.push(...groupLinks);
    const origin = drag?.[GROUP_KEY + groupId] ?? {
      x: group.rect[0],
      y: group.rect[1],
    };
    const saved = savedPositions(group);
    const generate = groupGenerate(graph, groupId);
    const genPos = generate
      ? (drag?.[generate.id] ?? { x: generate.pos[0], y: generate.pos[1] })
      : { x: origin.x + 40, y: origin.y + 60 };
    let nextY =
      genPos.y + (generate ? (sizes.get(generate.id)?.height ?? 120) : 0) + CARD_GAP;
    let bottom = 0;
    for (const card of cards) {
      let pos = drag?.[card.id];
      if (!pos) {
        const rel = saved[card.id.slice(AUX_PREFIX.length)];
        if (Array.isArray(rel)) pos = { x: origin.x + rel[0], y: origin.y + rel[1] };
      }
      if (!pos) {
        pos = { x: genPos.x, y: nextY };
        nextY += (sizes.get(card.id)?.height ?? CARD_HEIGHT) + CARD_GAP / 2;
      }
      elements.push({ card, x: pos.x, y: pos.y });
      bottom = Math.max(bottom, pos.y + (sizes.get(card.id)?.height ?? CARD_HEIGHT));
    }
    const needed = bottom - origin.y + CARD_GAP;
    if (needed > group.rect[3]) heights[groupId] = Math.round(needed);
  }
  return { elements, links, heights };
}

/** The group with a dragged card's spot saved relative to the frame. */
export function withAuxPosition(
  group: FlowGroup,
  cardId: string,
  pos: [number, number],
): FlowGroup {
  const saved = savedPositions(group);
  return {
    ...group,
    ui: {
      ...(group.ui ?? {}),
      [AUX_UI_KEY]: {
        ...saved,
        [cardId.slice(AUX_PREFIX.length)]: [
          Math.round(pos[0] - group.rect[0]),
          Math.round(pos[1] - group.rect[1]),
        ],
      },
    },
  };
}

/** The group id inside a card id (`recipe-aux:<group>:<kind>:<name>`). */
export function auxGroupId(cardId: string): string {
  return cardId.slice(AUX_PREFIX.length).split(":")[0];
}
