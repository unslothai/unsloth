// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DocGraph } from "../stores/doc-store";
import type { FlowGroup, FlowNote } from "../types";
import { isRecipeGroup } from "./recipe-columns";

/** The recipe group a note came from: import names it `<group id>_<Recipe Studio note id>`. */
export function noteRecipeGroup(
  graph: DocGraph,
  noteId: string,
): FlowGroup | null {
  for (const id of graph.groupOrder) {
    const group = graph.groups[id];
    if (isRecipeGroup(group) && noteId.startsWith(`${group.id}_`)) return group;
  }
  return null;
}

/** The group with its `ui.nodes` markdown notes rebuilt from the flow's notes, so the recipe keeps them. */
export function withRecipeNotes(
  group: FlowGroup,
  notes: FlowNote[],
): FlowGroup {
  const ui = group.ui ?? {};
  const prior = new Map(
    (Array.isArray(ui.nodes) ? ui.nodes : [])
      .filter((n): n is Record<string, unknown> => !!n && typeof n === "object")
      .map((n) => [String(n.id), n]),
  );
  const prefix = `${group.id}_`;
  const nodes = notes
    .filter((note) => note.id.startsWith(prefix))
    .map((note) => {
      const id = note.id.slice(prefix.length);
      return {
        ...(prior.get(id) ?? {}),
        id,
        name: String(prior.get(id)?.name ?? id),
        // biome-ignore lint/style/useNamingConvention: api schema
        node_type: "markdown_note",
        markdown: note.text,
        // biome-ignore lint/style/useNamingConvention: api schema
        ...(note.color ? { note_color: note.color } : {}),
      };
    });
  return { ...group, ui: { ...ui, nodes } };
}
