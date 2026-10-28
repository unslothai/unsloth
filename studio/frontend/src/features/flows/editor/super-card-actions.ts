// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useDocStore } from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";
import { GROUP_PREFIX } from "./derive-rf";
import { withSuperCard } from "./super-cards";
import { groupSelection } from "./use-editor-shortcuts";

export function setSuperCard(groupId: string, collapsed: boolean): void {
  const doc = useDocStore.getState();
  if (!doc.graph.groups[groupId]) return;
  doc.setMeta({ editor: withSuperCard(doc.meta.editor, groupId, collapsed) });
  const ui = useUiStore.getState();
  ui.setSelection([GROUP_PREFIX + groupId], []);
}

/** Collapse the selected group, or group the selected blocks and collapse that. */
export function makeSuperCard(): string | null {
  const ui = useUiStore.getState();
  const frames = ui.selectedNodes.filter((id) => id.startsWith(GROUP_PREFIX));
  const graph = useDocStore.getState().graph;
  const blocks = ui.selectedNodes.filter((id) => graph.nodes[id]);
  let groupId: string | undefined;
  if (frames.length === 1 && !blocks.length) {
    groupId = frames[0].slice(GROUP_PREFIX.length);
  } else if (blocks.length) {
    const before = new Set(graph.groupOrder);
    groupSelection();
    groupId = useDocStore
      .getState()
      .graph.groupOrder.find((id) => !before.has(id));
  }
  if (!groupId) return null;
  setSuperCard(groupId, true);
  return groupId;
}
