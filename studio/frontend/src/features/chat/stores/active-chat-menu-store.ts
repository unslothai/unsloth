// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { SidebarItem } from "../hooks/use-chat-sidebar-items";

type Place = { id: string; name: string };

export interface ActiveChatMenu {
  item: SidebarItem;
  pinned: boolean;
  unread: boolean;
  canFork: boolean;
  projects: Place[];
  sections: Place[];
  project: Place | null;
  section: Place | null;
  rename: () => void;
  togglePin: () => void;
  toggleUnread: () => void;
  fork: () => void;
  moveToProject: (id: string | null) => void;
  newProject: () => void;
  moveToSection: (id: string | null) => void;
  newSection: () => void;
  copyMarkdown: () => void;
  copySessionId: () => void;
  archive: () => void;
  remove: () => void;
}

export const useActiveChatMenuStore = create<{ menu: ActiveChatMenu | null }>(() => ({
  menu: null,
}));
