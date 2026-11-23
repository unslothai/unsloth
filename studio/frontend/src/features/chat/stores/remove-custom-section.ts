// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type SidebarCustomSection,
  assignmentMap,
  customSectionScope,
  inSectionOrder,
  resolveSectionOrder,
  useSidebarOrganizationStore,
} from "./sidebar-organization-store";

export function removeCustomSectionWithUndo(section: SidebarCustomSection): () => void {
  const state = useSidebarOrganizationStore.getState();
  const index = state.customSections.findIndex((s) => s.id === section.id);
  const chatIds = Object.keys(state.sectionByChatId).filter(
    (id) => state.sectionByChatId[id] === section.id,
  );
  const projectIds = Object.keys(state.sectionByProjectId).filter(
    (id) => state.sectionByProjectId[id] === section.id,
  );
  const pageIds = Object.keys(state.sectionByPageId).filter(
    (id) => state.sectionByPageId[id] === section.id,
  );
  const order = state.manualOrder[customSectionScope(section.id)];
  const hidden = state.hiddenSections.includes(section.id);
  const drawnOrder = resolveSectionOrder(state.sectionOrder, state.customSections);
  const followers = drawnOrder.slice(drawnOrder.indexOf(section.id) + 1);
  state.deleteCustomSection(section.id);
  return () => {
    useSidebarOrganizationStore.setState((now) => {
      if (now.customSections.some((s) => s.id === section.id)) return now;
      const restored = [...now.customSections];
      restored.splice(Math.min(index, restored.length), 0, section);
      const sectionByChatId = assignmentMap(now.sectionByChatId);
      for (const id of chatIds) sectionByChatId[id] ??= section.id;
      const sectionByProjectId = assignmentMap(now.sectionByProjectId);
      for (const id of projectIds) sectionByProjectId[id] ??= section.id;
      const sectionByPageId = assignmentMap(now.sectionByPageId);
      for (const id of pageIds) sectionByPageId[id] ??= section.id;
      const sectionOrder = resolveSectionOrder(now.sectionOrder, now.customSections);
      const follower = followers.find((key) => sectionOrder.includes(key));
      sectionOrder.splice(
        follower === undefined ? sectionOrder.length : sectionOrder.indexOf(follower),
        0,
        section.id,
      );
      return {
        customSections: inSectionOrder(restored, sectionOrder),
        sectionOrder,
        sectionByChatId,
        sectionByProjectId,
        sectionByPageId,
        hiddenSections: hidden
          ? [...now.hiddenSections, section.id]
          : now.hiddenSections,
        manualOrder: order
          ? { ...now.manualOrder, [customSectionScope(section.id)]: order }
          : now.manualOrder,
      };
    });
  };
}
