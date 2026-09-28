// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatProjects, useChatSidebarItems, useSidebarOrganizationStore } from "@/features/chat";
import { useMemo } from "react";
import { useChatFavoritesStore } from "./favorites-store";
import { matchesTerms, searchTerms } from "./model";

/** Starred chats, projects and sections Favorites lists for `query`, excluding archived chats. */
export function useFavoriteChatMatches(query: string, enabled: boolean): number {
  const { items } = useChatSidebarItems({ enabled, requireMessages: false });
  const { projects } = useChatProjects();
  const sections = useSidebarOrganizationStore((s) => s.customSections);
  const sectionByChatId = useSidebarOrganizationStore((s) => s.sectionByChatId);
  const chatIds = useChatFavoritesStore((s) => s.chatIds);
  const projectIds = useChatFavoritesStore((s) => s.projectIds);
  const sectionIds = useChatFavoritesStore((s) => s.sectionIds);
  return useMemo(() => {
    if (!enabled || chatIds.length + projectIds.length + sectionIds.length === 0) return 0;
    const terms = searchTerms(query);
    const starredChats = new Set(chatIds);
    const starredProjects = new Set(projectIds);
    const starredSections = new Set(sectionIds);
    const names = new Map(projects.map((project) => [project.id, project.name]));
    const sectionNames = new Map(sections.map((section) => [section.id, section.name]));
    const chats = items.filter((chat) => {
      if (!starredChats.has(chat.id)) return false;
      const sectionId = sectionByChatId[chat.id];
      return matchesTerms(
        terms,
        chat.title,
        chat.projectId ? names.get(chat.projectId) : undefined,
        sectionId ? sectionNames.get(sectionId) : undefined,
        ...(chat.modelIds ?? []),
      );
    }).length;
    const folders = projects.filter(
      (project) =>
        starredProjects.has(project.id) && matchesTerms(terms, project.name, project.instructions),
    ).length;
    const filed = sections.filter(
      (section) => starredSections.has(section.id) && matchesTerms(terms, section.name),
    ).length;
    return chats + folders + filed;
  }, [enabled, query, items, projects, sections, sectionByChatId, chatIds, projectIds, sectionIds]);
}
