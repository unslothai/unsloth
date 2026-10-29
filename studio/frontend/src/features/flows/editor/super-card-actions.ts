// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useDocStore } from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";
import {
  SUPER_PREFIX,
  type SuperCard,
  combineSuperCard,
  readSuperCards,
  updateSuperCard,
} from "./super-cards";

function nextId(cards: SuperCard[]): string {
  let max = 0;
  for (const { id } of cards) {
    const match = /^sc(\d+)$/.exec(id);
    if (match) max = Math.max(max, Number(match[1]));
  }
  return `sc${max + 1}`;
}

/** Combine the selected blocks into one closed super card and select it. */
export function combineSelection(): string | null {
  const doc = useDocStore.getState();
  const ui = useUiStore.getState();
  const id = nextId(readSuperCards(doc.meta.editor, doc.graph));
  const result = combineSuperCard(
    doc.meta.editor,
    doc.graph,
    ui.selectedNodes,
    id,
  );
  if (!result) return null;
  doc.setMeta({ editor: result.editor });
  ui.setSelection([SUPER_PREFIX + id], []);
  return id;
}

export function setSuperCardOpen(id: string, open: boolean): void {
  const doc = useDocStore.getState();
  const card = readSuperCards(doc.meta.editor, doc.graph).find(
    (c) => c.id === id,
  );
  if (!card) return;
  doc.setMeta({
    editor: updateSuperCard(doc.meta.editor, doc.graph, id, { open }),
  });
  useUiStore
    .getState()
    .setSelection(open ? card.members : [SUPER_PREFIX + id], []);
}

/** Drop the card; its blocks stay exactly where they are. */
export function ungroupSuperCard(id: string): void {
  const doc = useDocStore.getState();
  const card = readSuperCards(doc.meta.editor, doc.graph).find(
    (c) => c.id === id,
  );
  if (!card) return;
  doc.setMeta({
    editor: updateSuperCard(doc.meta.editor, doc.graph, id, null),
  });
  useUiStore.getState().setSelection(card.members, []);
}
