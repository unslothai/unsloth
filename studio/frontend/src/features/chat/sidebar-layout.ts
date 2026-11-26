// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  assignmentMap, customSectionIdOf, customSectionScope, inSectionOrder, resolveSectionOrder,
  useSidebarOrganizationStore, type SidebarCustomSection, type SidebarOrganizationState,
} from "./stores/sidebar-organization-store.ts";

export interface SidebarLayout {
  sections: Record<string, Omit<SidebarCustomSection, "id">>;
  sectionByChatId: Record<string, string>;
  sectionByProjectId: Record<string, string>;
  sectionByPageId: Record<string, string>;
  rowOrder: Record<string, string[]>;
  sectionOrder: string[];
}

export function sidebarLayout(state: SidebarOrganizationState): SidebarLayout {
  return {
    sections: Object.fromEntries(state.customSections.map(({ id, ...section }) => [id, section])),
    sectionByChatId: state.sectionByChatId,
    sectionByProjectId: state.sectionByProjectId,
    sectionByPageId: state.sectionByPageId,
    rowOrder: Object.fromEntries(state.customSections.flatMap(({ id }) => {
      const order = state.manualOrder[customSectionScope(id)];
      return order ? [[id, order]] : [];
    })),
    sectionOrder: resolveSectionOrder(state.sectionOrder, state.customSections),
  };
}

export function applySidebarLayout(state: SidebarOrganizationState, layout: SidebarLayout) {
  const sections = Object.entries(layout.sections).map(([id, section]) => ({ ...section, id }));
  const sectionOrder = resolveSectionOrder(layout.sectionOrder, sections);
  const manualOrder = Object.fromEntries(
    Object.entries(state.manualOrder).filter(([scope]) => customSectionIdOf(scope) === null),
  );
  for (const [id, ids] of Object.entries(layout.rowOrder)) manualOrder[customSectionScope(id)] = ids;
  return {
    customSections: inSectionOrder(sections, sectionOrder),
    sectionOrder,
    manualOrder,
    sectionByChatId: assignmentMap(layout.sectionByChatId),
    sectionByProjectId: assignmentMap(layout.sectionByProjectId),
    sectionByPageId: assignmentMap(layout.sectionByPageId),
  };
}

// A storage event rehydrates a peer's in-progress view, sometimes between creating a section
// and filing its first chat. Only the tab making those changes should submit them.
export function observeSidebarLayout(
  onChange: (fromStorage: boolean) => void,
): () => void {
  let hydrating = false;
  const store = useSidebarOrganizationStore;
  const start = store.persist.onHydrate(() => { hydrating = true; });
  const finish = store.persist.onFinishHydration(() => { hydrating = false; });
  const unsubscribe = store.subscribe((state, previous) => {
    if (!hydrating && state.syncedLayout !== previous.syncedLayout) return;
    const placedAfterUnpin = state.pendingLayoutEdit?.id === previous.pendingLayoutEdit?.id &&
      state.orderMoves?.slice(previous.orderMoves?.length ?? 0).some((move) => move.afterUnpin);
    if (
      !placedAfterUnpin && state.customSections === previous.customSections &&
      state.sectionByChatId === previous.sectionByChatId &&
      state.sectionByProjectId === previous.sectionByProjectId &&
      state.sectionByPageId === previous.sectionByPageId &&
      state.manualOrder === previous.manualOrder &&
      state.sectionOrder === previous.sectionOrder
    ) return;
    // Rehydration also allocates fresh arrays and maps for unchanged values.
    if (placedAfterUnpin || JSON.stringify(sidebarLayout(state)) !== JSON.stringify(sidebarLayout(previous))) {
      onChange(hydrating);
    }
  });
  return () => {
    start();
    finish();
    unsubscribe();
  };
}
