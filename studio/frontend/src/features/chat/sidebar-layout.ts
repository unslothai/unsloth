// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  assignmentMap, customSectionIdOf, customSectionScope, inSectionOrder, resolveSectionOrder,
  type SidebarCustomSection, type SidebarOrganizationState,
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
