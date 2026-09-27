// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatProjects, useChatSidebarItems } from "@/features/chat";
import { useMemo } from "react";
import { useChatFavoritesStore } from "./favorites-store";
import { matchesTerms, searchTerms } from "./model";

/** Starred chats and projects Favorites lists for `query`, excluding archived chats. */
export function useFavoriteChatMatches(query: string, enabled: boolean): number {
  const { items } = useChatSidebarItems();
  const { projects } = useChatProjects();
  const chatIds = useChatFavoritesStore((s) => s.chatIds);
  const projectIds = useChatFavoritesStore((s) => s.projectIds);
  return useMemo(() => {
    if (!enabled || (chatIds.length === 0 && projectIds.length === 0)) return 0;
    const terms = searchTerms(query);
    const starredChats = new Set(chatIds);
    const starredProjects = new Set(projectIds);
    const names = new Map(projects.map((project) => [project.id, project.name]));
    const chats = items.filter(
      (chat) =>
        starredChats.has(chat.id) &&
        matchesTerms(terms, chat.title, chat.projectId ? names.get(chat.projectId) : undefined),
    ).length;
    const folders = projects.filter(
      (project) =>
        starredProjects.has(project.id) && matchesTerms(terms, project.name, project.instructions),
    ).length;
    return chats + folders;
  }, [enabled, query, items, projects, chatIds, projectIds]);
}
